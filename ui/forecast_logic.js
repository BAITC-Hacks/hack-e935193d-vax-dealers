"use strict";

// Pure data functions: shared by the page and the offline Node.js tests.
const ForecastView = (() => {
  const fields = [
    "issue_date",
    "valid_local",
    "turbine_id",
    "lead_hours",
    "prediction",
    "lower90",
    "upper90",
    "wind",
    "temp",
    "status",
  ];

  function selectRows(rows, issueDate, turbine, horizon) {
    const hours = Number(horizon);
    const id = turbine === "all" ? "all" : Number(turbine);
    if (![24, 48].includes(hours) || !["all", 1, 2].includes(id)) {
      throw new Error("Выберите турбину 1/2 и горизонт 24/48 часов");
    }
    return rows
      .filter(
        (row) =>
          row.issue_date === issueDate &&
          (id === "all" || row.turbine_id === id) &&
          row.lead_hours <= hours,
      )
      .sort(
        (a, b) => a.lead_hours - b.lead_hours || a.turbine_id - b.turbine_id,
      );
  }

  function toCsv(rows) {
    const quote = (value) =>
      '"' + String(value ?? "").replaceAll('"', '""') + '"';
    return (
      fields.join(",") +
      "\r\n" +
      rows
        .map((row) => fields.map((key) => quote(row[key])).join(","))
        .join("\r\n")
    );
  }

  return { selectRows, toCsv };
})();

if (typeof module !== "undefined") module.exports = ForecastView;
