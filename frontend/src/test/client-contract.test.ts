/**
 * Request-contract tests: each API client function must emit exactly the
 * request the backend expects. Catches the payload-dropping class of bug
 * at the wire level.
 */
import { afterEach, describe, expect, it, vi } from "vitest";

import {
  cancelJob,
  fillMissingBulk,
  pauseAllJobs,
  pauseJob,
  resumeAllJobs,
  scanWatchNow,
  updateSchedule,
  updateWatch,
  uploadPdf,
} from "../api/client";

type Captured = { url: string; method: string; body: unknown };
const captured: Captured[] = [];

function mockFetch() {
  captured.length = 0;
  vi.stubGlobal(
    "fetch",
    vi.fn(async (url: string, opts: RequestInit = {}) => {
      let body: unknown = opts.body;
      if (typeof body === "string") body = JSON.parse(body);
      captured.push({ url: String(url), method: opts.method || "GET", body });
      return new Response(JSON.stringify({ success: true, data: {} }), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      });
    })
  );
}

afterEach(() => vi.unstubAllGlobals());

describe("API client request contracts", () => {
  it("fillMissingBulk forwards every option including priority", async () => {
    mockFetch();
    await fillMissingBulk({
      doc_ids: ["d1"],
      text: false,
      visual: false,
      entities: true,
      recover_text: true,
      priority: true,
    });
    expect(captured[0].url).toBe("/admin/fill-missing");
    expect(captured[0].body).toEqual({
      doc_ids: ["d1"],
      text: false,
      visual: false,
      entities: true,
      recover_text: true,
      priority: true,
    });
  });

  it("uploadPdf appends the priority form field only when requested", async () => {
    mockFetch();
    const f = new File(["%PDF"], "a.pdf", { type: "application/pdf" });
    await uploadPdf(f, "default", [], [], true);
    let fd = captured[0].body as FormData;
    expect(captured[0].url).toBe("/ingest");
    expect(fd.get("priority")).toBe("true");

    await uploadPdf(f, "default", [], []);
    fd = captured[1].body as FormData;
    expect(fd.get("priority")).toBeNull();
  });

  it("job controls hit the right endpoints with POST", async () => {
    mockFetch();
    await pauseAllJobs();
    await resumeAllJobs();
    await pauseJob("j1");
    await cancelJob("j1");
    await scanWatchNow();
    expect(captured.map((c) => [c.method, c.url])).toEqual([
      ["POST", "/ingest/jobs/pause-all"],
      ["POST", "/ingest/jobs/resume-all"],
      ["POST", "/ingest/jobs/j1/pause"],
      ["POST", "/ingest/jobs/j1/cancel"],
      ["POST", "/schedule/watch/scan-now"],
    ]);
  });

  it("schedule updates PUT the patch bodies verbatim", async () => {
    mockFetch();
    await updateSchedule({ enabled: true, start: "21:00", end: "06:30", days: [0, 6] });
    await updateWatch({ enabled: true, path: "/x", collection: "default" });
    expect(captured[0].method).toBe("PUT");
    expect(captured[0].body).toEqual({
      enabled: true, start: "21:00", end: "06:30", days: [0, 6],
    });
    expect(captured[1].url).toBe("/schedule/watch");
    expect(captured[1].body).toEqual({
      enabled: true, path: "/x", collection: "default",
    });
  });
});
