// These test data selection/export, not browser rendering.
const assert = require("node:assert/strict");
const view = require("../ui/forecast_logic.js");
const rows = [];
for (const date of ["2026-01-31", "2026-02-28"])
  for (let lead = 48; lead >= 1; lead--)
    for (const tid of [2, 1])
      rows.push({
        issue_date: date,
        lead_hours: lead,
        turbine_id: tid,
        valid_local: "2026-02-01 00:00:00",
        prediction: 0.123456789,
        lower90: 0,
        upper90: 0.5,
        wind: null,
        temp: -2.5,
        status: "weather_model",
      });
const before = JSON.stringify(rows);
assert.equal(view.selectRows(rows, "2026-01-31", "all", 48).length, 96);
const selected = view.selectRows(rows, "2026-02-28", "2", 24);
assert.equal(selected.length, 24);
assert.deepEqual(
  selected.map((row) => row.lead_hours),
  Array.from({ length: 24 }, (_, i) => i + 1),
);
assert(
  selected.every(
    (row) => row.turbine_id === 2 && row.issue_date === "2026-02-28",
  ),
);
assert.equal(view.selectRows(rows, "1900-01-01", "all", 24).length, 0);
assert.equal(
  JSON.stringify(rows),
  before,
  "Selection must not mutate the source data",
);
assert.throws(() => view.selectRows(rows, "2026-01-31", "3", 48));
assert.throws(() => view.selectRows(rows, "2026-01-31", "all", 72));
const csv = view.toCsv(selected);
assert.equal(csv.split("\r\n").length, 25);
assert(csv.includes('"0.123456789"'), "CSV must preserve precision");
assert(csv.includes('"","-2.5"'), "Missing wind must remain empty, not zero");
assert(view.toCsv([{ status: 'a,"b"' }]).includes('"a,""b"""'));
console.log(
  "Dashboard: filtering, dates, sorting, CSV precision/escaping and missing values passed.",
);
