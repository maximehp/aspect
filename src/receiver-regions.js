import { conicImports } from './silhouette.js';
// Shared by all receiver workers within one pose. Only storage and the compiled
// mathematical kernel survive an edit. Every entry is regenerated from geometry.
const module = WebAssembly.compileStreaming(fetch(new URL('./function-transport.wasm', import.meta.url)));
export class ReceiverRegions {
  async build(packet, tolerance) {
    this.packet = packet;
    this.wasm ??= new WebAssembly.Instance(await module, {
      math: Math,
      geometry: conicImports(() => ({ packet: this.packet, memory: this.wasm.memory })),
    }).exports;
    const { shapes, tiles, domains, lamp } = packet,
      nt = tiles.byteLength / 168,
      ng = domains.byteLength / 104,
      ns = shapes.length / 19;
    let offset = Number(this.wasm.__heap_base);
    function reserve(size) {
      const start = offset;
      offset += Math.ceil(size / 8) * 8;
      return start;
    }
    const s = reserve(shapes.byteLength),
      t = reserve(tiles.byteLength),
      g = reserve(domains.byteLength),
      l = reserve(24);
    const m = reserve(nt * ng),
      v = reserve(nt * 48),
      counts = reserve(160),
      ids = reserve(4 * nt * (ns + 1));
    const memory = this.wasm.memory;
    if (memory.buffer.byteLength < offset)
      memory.grow(Math.ceil((offset - memory.buffer.byteLength) / 65536));
    const buffer = memory.buffer;
    new Float64Array(buffer, s, shapes.length).set(shapes);
    new Uint8Array(buffer, t, tiles.byteLength).set(new Uint8Array(tiles));
    new Uint8Array(buffer, g, domains.byteLength).set(new Uint8Array(domains));
    new Float64Array(buffer, l, 3).set(lamp);
    new Uint8Array(buffer, counts, 160).fill(0);
    this.wasm.buildRegions(s, ns, t, nt, g, ng, l, tolerance, m, v, counts, ids);
    packet.regions = new Uint8Array(buffer, m, nt * ng).slice();
    packet.moments = new Float64Array(buffer, v, nt * 6).slice();
    packet.lampIds = new Int32Array(buffer, ids, nt * (ns + 1)).slice();
    const address = offset;
    let capacity = nt * 12;
    for (;;) {
      const end = address + capacity * 8;
      if (memory.buffer.byteLength < end) memory.grow(Math.ceil((end - memory.buffer.byteLength) / 65536));
      const used = this.wasm.compileLampEvents(s, ns, t, nt, l, ids, address, capacity);
      if (used <= capacity) {
        packet.lampEvents = new Float64Array(memory.buffer, address, used).slice();
        break;
      }
      capacity = used;
    }
    this.packet = null;
    return Array.from(new BigInt64Array(memory.buffer, counts, 20), Number);
  }
}
