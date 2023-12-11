module.exports = {
  async up(db, client) {
    db.createCollection(
      "pose_track_3d_pose_3d_links",
    );
    db.createCollection(
      "pose_track_3d_person_links",
    );
  },

  async down(db, client) {
    db.collection("pose_track_3d_person_links").drop();
    db.collection("pose_track_3d_pose_3d_links").drop();
  }
};
