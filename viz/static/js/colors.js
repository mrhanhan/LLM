export function divergingRGB(t) {
  t = Math.max(-1, Math.min(1, t));
  if (t >= 0) return [0.16 + 0.84 * t, 0.08 + 0.16 * t, 0.08 + 0.16 * t];
  const a = Math.abs(t);
  return [0.08 + 0.16 * a, 0.16 + 0.28 * a, 0.16 + 0.84 * a];
}
export function colorFor(t) {
  const [r, g, b] = divergingRGB(t);
  return { r, g, b };
}
