// Handles dynamic charts for the Admin Dashboard
document.addEventListener("DOMContentLoaded", function () {
  fetch("/dashboard_data")
    .then((res) => res.json())
    .then((data) => {
      // Category distribution pie chart
      const cat = data.category_distribution || {};
      const labels = Object.keys(cat);
      const vals = Object.values(cat);

      Plotly.newPlot(
        "category_dist",
        [{ values: vals, labels: labels, type: "pie", hole: 0.4 }],
        { title: "Allocated Category Distribution" }
      );

      // NDCG@5 metric bar chart
      Plotly.newPlot(
        "ndcg_chart",
        [{ x: ["NDCG@5"], y: [data.ndcg_mean], type: "bar", marker: { color: "#007bff" } }],
        { title: "Mean Ranking Quality (NDCG@5)", yaxis: { range: [0, 1] } }
      );
    })
    .catch((err) => {
      console.error("Error loading dashboard data:", err);
    });
});
