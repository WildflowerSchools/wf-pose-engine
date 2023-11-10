module.exports = {
  async up(db, client) {
    db.createCollection(
      "poses_2d",
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
    db.collection("poses_2d").drop()
  }
};
