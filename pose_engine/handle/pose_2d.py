from pymongo import MongoClient

import pose_engine.config


class Pose2d:
    def __init__(self, db_uri: str = None):
        if db_uri is None:
            db_uri = pose_engine.config.Settings()

        self.client = MongoClient(db_uri)
        self.db = self.client["poses_2d"]

    def insert_poses(init):
        # TODO: Probably use pydantic-mongo object mapping tool to handle the creation of pose records
        pass
