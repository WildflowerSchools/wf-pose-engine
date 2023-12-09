module.exports = {
  async up(db, client) {
    db.createCollection(
      "pose_tracks_3d",
    )
  },

  async down(db, client) {
    db.collection("pose_tracks_3d").drop()
  }
};
