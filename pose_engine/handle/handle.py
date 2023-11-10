# from bson.binary import UuidRepresentation
from typing import List

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

        self.client: MongoClient = MongoClient(db_uri, uuidRepresentation="standard")
        self.db: MongoDatabase = self.client["poses"]
        self.poses_collection: MongoCollection = self.db["poses_2d"]

    def insert_poses(self, pose_2d_batch: List[Pose2d]):
        bulk_requests = list(map(lambda p: InsertOne(p.model_dump()), pose_2d_batch))
        # TODO: Probably use pydantic-mongo object mapping tool to handle the creation of pose records
        try:
            result = self.poses_collection.bulk_write(bulk_requests, ordered=False)
        except BulkWriteError as e:
            logger.error(f"Failed writing pose_2d_batch to Mongo: {e}")
