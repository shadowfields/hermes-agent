import type { AuxiliaryTaskAssignment } from "./api";

export interface AuxiliaryTaskMetadata {
  key: string;
  label: string;
  hint: string;
}

const BUILTIN_AUXILIARY_TASKS: readonly AuxiliaryTaskMetadata[] = [
  { key: "vision", label: "Vision", hint: "Image analysis" },
  { key: "compression", label: "Compression", hint: "Context compaction" },
  { key: "skills_hub", label: "Skills Hub", hint: "Skill search" },
  { key: "approval", label: "Approval", hint: "Smart auto-approve" },
  { key: "mcp", label: "MCP", hint: "MCP tool routing" },
  { key: "title_generation", label: "Title Gen", hint: "Session titles" },
  { key: "review", label: "Review", hint: "/review subagent" },
  { key: "triage_specifier", label: "Triage Specifier", hint: "Kanban spec fleshing" },
  { key: "kanban_decomposer", label: "Kanban Decomposer", hint: "Task decomposition" },
  { key: "profile_describer", label: "Profile Describer", hint: "Auto profile descriptions" },
  { key: "curator", label: "Curator", hint: "Skill-usage review" },
];

function fallbackLabel(key: string): string {
  const words = key.replaceAll("_", " ");
  return words.charAt(0).toUpperCase() + words.slice(1);
}

/** Preserve the built-in presentation while appending backend-discovered plugin tasks. */
export function resolveAuxiliaryTaskMetadata(
  assignments: readonly AuxiliaryTaskAssignment[] = [],
): AuxiliaryTaskMetadata[] {
  const tasks = [...BUILTIN_AUXILIARY_TASKS];
  const seen = new Set(tasks.map((task) => task.key));
  for (const assignment of assignments) {
    if (!assignment.task || seen.has(assignment.task)) continue;
    seen.add(assignment.task);
    tasks.push({
      key: assignment.task,
      label: assignment.display_name || fallbackLabel(assignment.task),
      hint: assignment.description || assignment.task,
    });
  }
  return tasks;
}
