from typing import List
from uuid import UUID

from pymongo import InsertOne, MongoClient
from pymongo.collection import Collection as MongoCollection
from pymongo.database import Database as MongoDatabase
from pymongo.errors import BulkWriteError

import pose_engine.config
from pose_engine.log import logger

from .models.pose_2d import Pose2d


class Pose2dHandle:
    def __init__(self, db_uri: str = None):
        if db_uri is None:
            db_uri = pose_engine.config.Settings().MONGO_POSE2D_URI

        self.client: MongoClient = MongoClient(
            db_uri, uuidRepresentation="standard", tz_aware=True
        )
        self.db: MongoDatabase = self.client.get_database("poses")
        self.poses_collection: MongoCollection = self.db.get_collection("poses_2d")

    def fetch_current_pose_coverage(self):
        poses_cursor = self.poses_collection.aggregate(
            [
                # {
                #     '$match': {
                #         'metadata.environment_id': UUID(environment_id)
                #     }
                # },
                {
                    "$group": {
                        "_id": {
                            "metadata": {
                                "inference_run_created_at": "$metadata.inference_run_created_at",
                                "inference_run_id": "$metadata.inference_run_id",
                            }
                        },
                        "" "inference_run_id": {"$first": "$metadata.inference_run_id"},
                        "inference_run_created_at": {
                            "$first": "$metadata.inference_run_created_at"
                        },
                        "environment_id": {"$first": "$metadata.environment_id"},
                        "count": {"$sum": 1},
                        "start": {"$min": "$timestamp"},
                        "end": {"$max": "$timestamp"},
                    }
                },
                {"$sort": {"start": -1, "inference_run_created_at": -1}},
            ]
        )
        return list(poses_cursor)

    def insert_poses(self, pose_2d_batch: List[Pose2d]):
        bulk_requests = list(map(lambda p: InsertOne(p.model_dump()), pose_2d_batch))
        try:
            logger.debug(
                f"Inserting {len(bulk_requests)} into Mongo poses_2d database..."
            )
            self.poses_collection.bulk_write(bulk_requests, ordered=False)
            logger.debug(
                f"Successfully wrote {len(bulk_requests)} records into Mongo poses_2d database..."
            )
        except BulkWriteError as e:
            logger.error(
                f"Failed writing {len(bulk_requests)} records to Mongo poses_2d database: {e}"
            )

    def cleanup(self):
        if self.client is not None:
            self.client.close()

    def __del__(self):
        self.cleanup()
