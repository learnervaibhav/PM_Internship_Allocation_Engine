document.addEventListener("DOMContentLoaded", function () {
  document.querySelectorAll(".results-table").forEach((table) => {
    table.querySelectorAll("tbody tr").forEach((row) => {
      row.addEventListener("click", () => {
        alert("Selected Internship ID: " + row.cells[1].textContent);
      });
    });
  });
});
