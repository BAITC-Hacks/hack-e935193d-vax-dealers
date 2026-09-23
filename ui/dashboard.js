"use strict";
const data = JSON.parse(document.getElementById("forecast-data").textContent);
const byId = (id) => document.getElementById(id);
const number = (v, digits = 3) =>
  v === null || !Number.isFinite(Number(v))
    ? "—"
    : Number(v).toLocaleString("ru-RU", {
        minimumFractionDigits: digits,
        maximumFractionDigits: digits,
      });
const dateLabel = (s) =>
  s.slice(8, 10) + "." + s.slice(5, 7) + "." + s.slice(0, 4);
const timeLabel = (s) => dateLabel(s) + " " + s.slice(11, 16);
const element = (tag, text, cls) => {
  const n = document.createElement(tag);
  if (text !== undefined) n.textContent = text;
  if (cls) n.className = cls;
  return n;
};
const issues = [...new Set(data.rows.map((r) => r.issue_date))].sort();
for (const issue of issues) {
  const o = element("option", dateLabel(issue));
  o.value = issue;
  byId("issue").append(o);
}
byId("timezone").textContent =
  "Время: UTC" +
  (data.summary.utc_offset_hours >= 0 ? "+" : "") +
  data.summary.utc_offset_hours +
  " (допущение)";
function selected() {
  return ForecastView.selectRows(
    data.rows,
    byId("issue").value,
    byId("turbine").value,
    byId("horizon").value,
  );
}
const NS = "http://www.w3.org/2000/svg";
function svg(tag, attributes, text) {
  const n = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attributes)) n.setAttribute(k, v);
  if (text !== undefined) n.textContent = text;
  return n;
}
function draw(rows) {
  const chart = byId("chart");
  chart.replaceChildren();
  const width = 1028,
    height = 264,
    left = 58,
    top = 22,
    leads = Number(byId("horizon").value);
  const x = (n) => left + ((n - 1) * width) / (leads - 1),
    y = (n) => top + (1 - n) * height;
  for (let i = 0; i <= 4; i++) {
    const value = i / 4;
    chart.append(
      svg("line", {
        x1: left,
        y1: y(value),
        x2: left + width,
        y2: y(value),
        stroke: "#e0e8e8",
      }),
    );
    chart.append(
      svg(
        "text",
        { x: left - 12, y: y(value) + 4, "text-anchor": "end" },
        number(value, 2),
      ),
    );
  }
  for (const lead of [
    ...new Set([
      1,
      Math.round(leads / 4),
      Math.round(leads / 2),
      Math.round((3 * leads) / 4),
      leads,
    ]),
  ]) {
    const r = rows.find((r) => r.lead_hours === lead);
    if (r) {
      chart.append(
        svg(
          "text",
          { x: x(lead), y: top + height + 26, "text-anchor": "middle" },
          r.valid_local.slice(11, 16),
        ),
      );
      chart.append(
        svg(
          "text",
          { x: x(lead), y: top + height + 46, "text-anchor": "middle" },
          r.valid_local.slice(8, 10) + "." + r.valid_local.slice(5, 7),
        ),
      );
    }
  }
  for (const tid of [1, 2]) {
    const part = rows.filter((r) => r.turbine_id === tid),
      color = tid === 1 ? "#087f69" : "#315fd8";
    if (!part.length) continue;
    const path = part
      .map((r, i) => (i ? "L" : "M") + x(r.lead_hours) + "," + y(r.prediction))
      .join(" ");
    const area =
      part
        .map((r, i) => (i ? "L" : "M") + x(r.lead_hours) + "," + y(r.upper90))
        .join(" ") +
      " " +
      part
        .slice()
        .reverse()
        .map((r) => "L" + x(r.lead_hours) + "," + y(r.lower90))
        .join(" ") +
      " Z";
    chart.append(svg("path", { d: area, fill: color, opacity: 0.075 }));
    chart.append(
      svg("path", {
        d: path,
        fill: "none",
        stroke: color,
        "stroke-width": 2.6,
        "stroke-linejoin": "round",
      }),
    );
    for (const r of part) {
      const point = svg("circle", {
        cx: x(r.lead_hours),
        cy: y(r.prediction),
        r: r.status === "fallback_climatology" ? 4 : 3,
        fill: color,
      });
      point.append(
        svg(
          "title",
          {},
          "Турбина " +
            tid +
            " · " +
            timeLabel(r.valid_local) +
            " · " +
            number(r.prediction),
        ),
      );
      chart.append(point);
    }
  }
}
function render() {
  const rows = selected();
  if (!rows.length) return;
  const mean = rows.reduce((s, r) => s + r.prediction, 0) / rows.length;
  byId("mean").textContent = number(mean);
  byId("range").textContent =
    number(Math.min(...rows.map((r) => r.prediction)), 2) +
    " / " +
    number(Math.max(...rows.map((r) => r.prediction)), 2);
  byId("count").textContent = number(rows.length, 0);
  byId("countnote").textContent =
    byId("horizon").value +
    " ч × " +
    (byId("turbine").value === "all" ? "2 турбины" : "1 турбина");
  byId("fallback").textContent = rows.filter(
    (r) => r.status !== "weather_model",
  ).length;
  byId("chart-period").textContent =
    timeLabel(rows[0].valid_local) + " — " + timeLabel(rows.at(-1).valid_local);
  byId("table-count").textContent = rows.length + " строк";
  draw(rows);
  byId("rows").replaceChildren();
  for (const r of rows) {
    const tr = element("tr");
    for (const v of [
      timeLabel(r.valid_local),
      r.turbine_id,
      number(r.prediction),
      number(r.lower90) + "–" + number(r.upper90),
      number(r.wind, 1),
    ])
      tr.append(element("td", String(v)));
    const td = element("td"),
      fallback = r.status !== "weather_model";
    td.append(
      element(
        "span",
        fallback ? "Резервный" : "По погоде",
        fallback ? "badge warn" : "badge",
      ),
    );
    tr.append(td);
    byId("rows").append(tr);
  }
}
for (const id of ["issue", "turbine", "horizon"])
  byId(id).addEventListener("change", render);
byId("download").addEventListener("click", () => {
  const csv = ForecastView.toCsv(selected());
  const url = URL.createObjectURL(
    new Blob(["\ufeff" + csv], { type: "text/csv;charset=utf-8" }),
  );
  const a = element("a");
  a.href = url;
  a.download =
    "forecast_" +
    byId("issue").value +
    "_" +
    byId("turbine").value +
    "_" +
    byId("horizon").value +
    "h.csv";
  document.body.append(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
});
const mt = element("table"),
  mh = element("thead"),
  hr = element("tr");
for (const v of ["Турбина", "Горизонт", "Модель", "Среднее", "Последнее"])
  hr.append(element("th", v));
mh.append(hr);
mt.append(mh);
const mb = element("tbody");
for (const tid of [1, 2])
  for (const horizon of ["1-24", "25-48"]) {
    const matches = data.metrics.filter(
      (m) => m.turbine_id === tid && m.horizon === horizon,
    );
    if (!matches.length) continue;
    const tr = element("tr");
    for (const v of [
      String(tid),
      horizon + " ч",
      ...["prediction", "climatology", "persistence"].map((name) =>
        number(matches.find((m) => m.model === name)?.MAE),
      ),
    ])
      tr.append(element("td", v));
    mb.append(tr);
  }
mt.append(mb);
byId("metrics").append(
  mb.children.length
    ? mt
    : element("p", "Метрики этой модели не включены в панель.", "empty"),
);
for (const [title, detail] of [
  [
    "Файлы и значения проверены",
    data.check.counts.forecast_rows + " прогнозов, без повторяющихся ключей",
  ],
  [
    "Февраль заполнен",
    data.check.counts.february_rows + " строк: 672 часа каждой турбины",
  ],
  [
    "Контрольные суммы совпадают",
    "Проверены исходные CSV результатов и манифест",
  ],
]) {
  const li = element("li");
  li.append(element("strong", title), element("small", detail));
  byId("checks").append(li);
}
for (const text of data.check.warnings)
  byId("assumptions").append(element("p", text));
for (const [tid, a] of Object.entries(data.audit))
  byId("audit").append(
    element(
      "p",
      "Турбина " +
        tid +
        ": " +
        number(a.rows, 0) +
        " измерений, " +
        number(a.complete_hours, 0) +
        " полных часов, " +
        number(a.missing_10min_slots, 0) +
        " отсутствующих десятиминутных слотов.",
    ),
  );
render();
