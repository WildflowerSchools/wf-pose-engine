from threading import Thread
import time
import queue

import torch.multiprocessing as mp
import torch.utils.data

from cv_utils import VideoInput

from pose_engine.log import logger


mp.set_start_method('spawn', force=True)


class VideoFramesDataset(torch.utils.data.IterableDataset):
    def __init__(self,
                 video_paths: list[str],
                 frame_queue_maxsize: int = 120,
                 wait_for_video_files: bool = True,
                 wait_for_video_frames: bool = True):


        # logger.debug(f"Initializing VideoFramesDataset using batch_size: {self.batch_size}")

        super(VideoFramesDataset).__init__()

        self.frame_queue_maxsize = frame_queue_maxsize
        self.wait_for_video_files = wait_for_video_files
        self.wait_for_video_frames = wait_for_video_frames

        self.video_path_queue = mp.JoinableQueue()
        for v in video_paths:
            self.video_path_queue.put(v)

        self.video_frame_queue = mp.Queue(maxsize=frame_queue_maxsize)
        
        self.video_loader_thread_stopped = False
        self.video_loader_thread = Thread(target=self.start_video_loader, args=())
        self.video_loader_thread.daemon = True
        self.video_loader_thread.start()

    def add_video_path(self, video_path):
        self.video_path_queue.put(video_path)

    def start_video_loader(self):
        while True:
            if self.video_loader_thread_stopped:
                break

            try:
                video_path = self.video_path_queue.get(block=False)
                
                if self.video_loader_thread_stopped:
                    break

                logger.info(f"Loading frames from {video_path}...")
                video_reader = VideoInput(input_path=video_path, queue_frames=True)

                frame_idx = 0
                while True:
                    frame = video_reader.get_frame()
                    if frame is None:
                        logger.debug(f"Exhausted all frames in {video_path}")
                        break
                    
                    frame_idx += 1

                    frame_written = False
                    while not frame_written:
                        try:
                            logger.debug(f"Putting {video_path} on frame queue")
                            self.video_frame_queue.put((frame, {"path": video_path, "frame_index": frame_idx}), timeout=2)
                            frame_written = True
                        except queue.Full:
                            logger.debug(f"Reached frame queue max size '{self.frame_queue_maxsize}', waiting to load additional frames...")
                            if self.video_loader_thread_stopped:
                                break

                self.video_path_queue.task_done()
            except queue.Empty:
                if not self.wait_for_video_files:
                    logger.info("Video file queue empty and wait_for_video_files set to False, killing video loader")
                    self.video_loader_thread_stopped = True
                
                time.sleep(1)
    
    def __iter__(self):
        while True:
            try:
                frame, meta = self.video_frame_queue.get(block=False, timeout=1)
                if frame is not None:
                    yield (frame, meta)
            except queue.Empty:
                # DO NOT REMOVE THE "qsize()" assertion, this is important as the queue.Empty exception doesn't necessarily mean the queue is empty
                if self.video_frame_queue.qsize() == 0:
                    if not self.wait_for_video_frames:
                        logger.info(f"Nothing to read from frame queue, terminating iterator")
                        break

                    if self.video_loader_thread_stopped:
                        logger.info(f"Video loader is terminated and no more items in the frame queue, terminating iterator")
                        break
                
                logger.debug(f"Nothing to read from frame queue, waiting...")
                # time.sleep(0.1)
    
    def stop_video_loader(self):
        self.video_loader_thread_stopped = True
        self.video_loader_thread.join()
