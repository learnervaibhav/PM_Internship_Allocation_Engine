document.addEventListener("DOMContentLoaded", function () {
  const table = document.getElementById("ranks_table");
  if (table) {
    // simple row highlighting
    table.querySelectorAll("tbody tr").forEach((row) => {
      row.addEventListener("mouseover", () => row.classList.add("highlight"));
      row.addEventListener("mouseout", () => row.classList.remove("highlight"));
    });
  }
});
