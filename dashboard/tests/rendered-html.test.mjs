import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

async function render() {
  const workerUrl = new URL("../dist/server/index.js", import.meta.url);
  workerUrl.searchParams.set("test", `${process.pid}-${Date.now()}`);
  const { default: worker } = await import(workerUrl.href);
  return worker.fetch(
    new Request("http://localhost/", { headers: { accept: "text/html" } }),
    { ASSETS: { fetch: async () => new Response("Not found", { status: 404 }) } },
    { waitUntil() {}, passThroughOnException() {} },
  );
}

test("server-renders the PV forecasting conference dashboard", async () => {
  const response = await render();
  assert.equal(response.status, 200);
  assert.match(response.headers.get("content-type") ?? "", /^text\/html\b/i);

  const html = await response.text();
  assert.match(html, /<title>Beyond the Model \| FEVER PV Forecasting<\/title>/i);
  assert.match(html, /THE FOURTH UK AI CONFERENCE 2026/);
  assert.match(html, /What if the biggest error in an AI forecast happens/);
  assert.match(html, /12 \/ 12 stages succeeded/);
  assert.match(html, /Built by Masood Nazari/);
  assert.doesNotMatch(html, /Your site is taking shape|codex-preview/);
});

test("ships a self-contained, data-safe standalone edition", async () => {
  const [standalone, page, ignore] = await Promise.all([
    readFile(new URL("../standalone/PV-Forecasting-Conference-Dashboard.html", import.meta.url), "utf8"),
    readFile(new URL("../app/page.tsx", import.meta.url), "utf8"),
    readFile(new URL("../.gitignore", import.meta.url), "utf8"),
  ]);

  assert.match(standalone, /^<!doctype html>/i);
  assert.match(standalone, /data:image\//);
  assert.match(standalone, /Built by Masood Nazari/);
  assert.doesNotMatch(standalone, /<script[^>]+src=/i);
  assert.doesNotMatch(standalone, /<link[^>]+stylesheet/i);
  assert.doesNotMatch(standalone, /PV_data\.csv|weather_cache\.csv/i);
  assert.match(page, /No confidential measurements embedded/);
  assert.match(ignore, /private_data/);
  assert.match(ignore, /camera_ready_outputs/);
  assert.match(ignore, /\*\.csv/);
});
