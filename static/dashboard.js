// Handles dynamic charts for the Admin Dashboard
document.addEventListener("DOMContentLoaded", function () {
  fetch("/dashboard_data")
    .then((res) => res.json())
    .then((data) => {
      // Category distribution pie chart
      const cat = data.category_distribution || {};
      const labels = Object.keys(cat);
      const vals = Object.values(cat);

      console.log('Eligible students:', eligible);
      console.log('Filtered out students:', filteredOut);

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

      // Student eligibility distribution pie chart
      function createPieChart(data) {
        const eligible = data.filter(item => item.eligible === 'Yes').length;
        const filteredOut = data.filter(item => item.eligible === 'No').length;
        
        const pieChartData = {
            labels: ['Eligible', 'Filtered Out'],
            datasets: [{
                data: [eligible, filteredOut],
                backgroundColor: ['#36A2EB', '#FF6384'],
                hoverBackgroundColor: ['#36A2EB', '#FF6384']
            }]
        };

        const ctx = document.getElementById('pieChart').getContext('2d');
        new Chart(ctx, {
            type: 'pie',
            data: pieChartData,
            options: {
                responsive: true,
                plugins: {
                    legend: {
                        position: 'top',
                    },
                    title: {
                        display: true,
                        text: 'Student Eligibility Distribution'
                    }
                }
            }
        });
      }

      createPieChart(data.student_data);
    })
    .catch((err) => {
      console.error("Error loading dashboard data:", err);
    });
});
