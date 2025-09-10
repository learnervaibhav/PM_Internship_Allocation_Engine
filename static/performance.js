document.addEventListener("DOMContentLoaded", function () {
  if (typeof graphJSON !== "undefined") {
    Plotly.newPlot("performance_graph", graphJSON.data, graphJSON.layout);
  }
});
