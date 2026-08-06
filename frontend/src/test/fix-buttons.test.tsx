/**
 * Contract tests for the audit fix panel and repair status display.
 *
 * These exist because of two shipped bugs: a mutation that rebuilt its
 * payload field-by-field silently dropped recover_text and priority (the
 * per-row "Recover OCR text" button ran without OCR recovery), and the
 * status display showed "done" for paused jobs. Every button here is
 * asserted against the exact options it must emit, and every job status
 * against the message the user must see.
 */
import { render, screen, cleanup } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";

import { DocFixButtons, RepairStatus } from "../pages/Manage";
import type { DocAudit } from "../api/client";
import type { JobRow } from "../api/types";

afterEach(cleanup);

type AspectStatus = "done" | "error" | "partial" | "missing" | "na";
const aspect = (status: AspectStatus, done = 0, needed = 0) => ({
  status,
  done,
  needed,
  detail: null,
});

const docWith = (over: Partial<DocAudit>): DocAudit =>
  ({
    doc_id: "d1",
    title: "Test Doc",
    collection: "default",
    source_type: "digital_native",
    pages: 10,
    declared_pages: 10,
    chunk_count: 5,
    recoverable_text_pages: 0,
    overall: "incomplete",
    aspects: {
      pages: aspect("done", 10, 10),
      text: aspect("done", 10, 10),
      text_embedding: aspect("done", 10, 10),
      visual_embedding: aspect("done", 10, 10),
      chunks: aspect("done", 10, 10),
      entities: aspect("done", 10, 10),
    },
    ...over,
  }) as DocAudit;

describe("DocFixButtons option forwarding", () => {
  it("forwards priority when ⚡ run immediately is checked", async () => {
    const onFill = vi.fn();
    const doc = docWith({
      aspects: {
        ...docWith({}).aspects,
        entities: aspect("partial", 4, 10),
      },
    });
    render(
      <DocFixButtons
        doc={doc}
        busy={false}
        queuedLabel={null}
        onFill={onFill}
        onChunks={() => {}}
        onReembed={() => {}}
      />
    );
    await userEvent.click(screen.getByRole("checkbox"));
    await userEvent.click(
      screen.getByRole("button", { name: /Extract missing entities/ })
    );
    expect(onFill).toHaveBeenCalledWith({
      text: false,
      visual: false,
      entities: true,
      priority: true,
    });
  });

  it("defaults to non-priority when the checkbox is untouched", async () => {
    const onFill = vi.fn();
    const doc = docWith({
      aspects: { ...docWith({}).aspects, entities: aspect("partial", 4, 10) },
    });
    render(
      <DocFixButtons
        doc={doc}
        busy={false}
        queuedLabel={null}
        onFill={onFill}
        onChunks={() => {}}
        onReembed={() => {}}
      />
    );
    await userEvent.click(
      screen.getByRole("button", { name: /Extract missing entities/ })
    );
    expect(onFill).toHaveBeenCalledWith(
      expect.objectContaining({ entities: true, priority: false })
    );
  });

  it("forwards recover_text on the OCR recovery button", async () => {
    const onFill = vi.fn();
    const doc = docWith({ recoverable_text_pages: 7 });
    render(
      <DocFixButtons
        doc={doc}
        busy={false}
        queuedLabel={null}
        onFill={onFill}
        onChunks={() => {}}
        onReembed={() => {}}
      />
    );
    await userEvent.click(
      screen.getByRole("button", { name: /Recover OCR text/ })
    );
    expect(onFill).toHaveBeenCalledWith(
      expect.objectContaining({ recover_text: true, entities: true })
    );
  });
});

describe("RepairStatus never lies", () => {
  const job = (status: JobRow["status"]): JobRow =>
    ({
      job_id: "j1",
      status,
      current_step: "extracting_entities",
      progress_pct: 40,
      error_message: null,
    }) as JobRow;

  const expectText = (status: JobRow["status"], pattern: RegExp) => {
    const { container, unmount } = render(
      <RepairStatus label="fix" job={job(status)} />
    );
    expect(container.textContent).toMatch(pattern);
    unmount();
  };

  it("paused shows held, not done", () => expectText("paused", /held/i));
  it("cancelled shows stopped, not done", () =>
    expectText("cancelled", /stopped/i));
  it("failed shows failed", () => expectText("failed", /failed/i));
  it("queued shows queued", () => expectText("queued", /queued/i));
  it("processing shows progress", () =>
    expectText("processing", /extracting_entities.*40/));
  it("only completed shows done", () => expectText("completed", /done/i));
});
