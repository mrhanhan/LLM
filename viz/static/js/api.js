const j = async (u) => (await fetch(u)).json();
const q = (o) => new URLSearchParams(o).toString();

export const getModel = (source = 'live') => j(`/api/model?${q({ source })}`);
export const getGrid = (name, source = 'live', tiles = 64) =>
  j(`/api/matrix/${encodeURIComponent(name)}/grid?${q({ source, tiles })}`);
export const getStats = (name, source = 'live') =>
  j(`/api/matrix/${encodeURIComponent(name)}/stats?${q({ source })}`);
export const getCell = (name, i, k, source = 'live') =>
  j(`/api/matrix/${encodeURIComponent(name)}/cell?${q({ source, i, j: k })}`);
export const getPatch = (name, r, c, h, w, source = 'live') =>
  j(`/api/matrix/${encodeURIComponent(name)}/patch?${q({ source, r, c, h, w })}`);
export const matrixPngUrl = (name, source = 'live', tiles = 64, norm = 'global') =>
  `/api/matrix/${encodeURIComponent(name)}/png?${q({ source, tiles, norm })}`;
export const getNeurons = (matrix, source = 'live', n = 160, top = 2500) =>
  j(`/api/neurons?${q({ matrix, source, n, top })}`);

export function connectWS(onmsg) {
  const ws = new WebSocket(`${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/ws`);
  ws.onmessage = (e) => onmsg(JSON.parse(e.data));
  ws.onclose = () => setTimeout(() => connectWS(onmsg), 1200);
  return ws;
}
