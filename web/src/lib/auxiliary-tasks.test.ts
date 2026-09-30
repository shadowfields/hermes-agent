import { describe, expect, it } from "vitest";

import { resolveAuxiliaryTaskMetadata } from "./auxiliary-tasks";

describe("resolveAuxiliaryTaskMetadata", () => {
  it("keeps built-in ordering and appends plugin tasks from backend discovery", () => {
    const tasks = resolveAuxiliaryTaskMetadata([
      { task: "vision", provider: "auto", model: "", base_url: "" },
      {
        task: "teams_summary",
        display_name: "Teams summary",
        description: "Grounded Microsoft Teams meeting summaries",
        provider: "auto",
        model: "",
        base_url: "",
      },
    ]);

    expect(tasks[0]).toMatchObject({ key: "vision", label: "Vision" });
    expect(tasks.at(-1)).toEqual({
      key: "teams_summary",
      label: "Teams summary",
      hint: "Grounded Microsoft Teams meeting summaries",
    });
    expect(tasks.filter((task) => task.key === "teams_summary")).toHaveLength(1);
  });
});
