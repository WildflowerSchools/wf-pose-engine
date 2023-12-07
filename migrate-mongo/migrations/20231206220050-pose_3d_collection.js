module.exports = {
  async up(db, client) {
    db.createCollection(
      "poses_3d",
      {
         timeseries: {
            timeField: "timestamp",
            metaField: "metadata",
            granularity: "seconds"
         }
      }
    )
  },

  async down(db, client) {
    db.collection("poses_3d").drop()
  }
};
