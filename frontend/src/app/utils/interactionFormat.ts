export function formatResponseTime(value?: string | null): string {
  if (!value) {
    return "N/A";
  }

  const trimmed = value.trim();
  if (!trimmed || trimmed.toUpperCase() === "N/A") {
    return "N/A";
  }

  return trimmed.endsWith("s") ? trimmed : `${trimmed}s`;
}

export function parseInteractionDateTime(date: string, time: string): number {
  const [year, month, day] = date.split("-").map(Number);
  if (!year || !month || !day) {
    return 0;
  }

  const trimmedTime = time.trim();
  const match = trimmedTime.match(/^(\d{1,2}):(\d{2})(?:\s*([AP]M))?$/i);
  if (!match) {
    return new Date(year, month - 1, day).getTime();
  }

  const rawHours = Number(match[1]);
  const minutes = Number(match[2]) || 0;
  const meridiem = match[3]?.toUpperCase();

  let hours = rawHours;
  if (meridiem) {
    hours = rawHours % 12;
    if (meridiem === "PM") {
      hours += 12;
    }
  }

  return new Date(year, month - 1, day, hours, minutes).getTime();
}
