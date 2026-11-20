import { DEFAULT_LAMP_RGB } from './lamp.js';
import { roundedBox } from './rounded-box.js';

// Coordinates are in metres. Each axis runs from the centre of a rectangle to
// the midpoint of one edge. Material colours are linear RGB reflectances.
const plaster = [0.82, 0.77, 0.66];
const limestone = [0.66, 0.60, 0.47];
const oak = [0.47, 0.26, 0.12];
const charcoal = [0.095, 0.11, 0.10];
const clay = [0.60, 0.155, 0.075];
const chalk = [0.92, 0.86, 0.69];
const blue = [0.095, 0.24, 0.29];
const lightPosition = [-5.5, 3.95, 1.2];

export const START_CAMERA = {
  position: [2.15, 1.62, 3.8],
  yaw: -0.38,
  pitch: -0.105,
  fov: Math.PI / 3,
};

export const WALK_BOUNDS = { minX: -3.16, maxX: 3.16, minZ: -4.16, maxZ: 4.26 };

function rectangle(id, center, axisU, axisV, color) {
  return { id, center, axisU, axisV, color };
}

function block(id, [x, y, z], [width, height, depth], color, caps = true) {
  const a = width / 2, b = height / 2, c = depth / 2;
  const X = [a, 0, 0], Y = [0, b, 0], Z = [0, 0, c];
  return [
    ...(caps ? [
      rectangle(id + '-top', [x, y + b, z], X, Z, color),
      rectangle(id + '-bottom', [x, y - b, z], X, Z, color),
    ] : []),
    rectangle(id + '-left', [x - a, y, z], Z, Y, color),
    rectangle(id + '-right', [x + a, y, z], Z, Y, color),
    rectangle(id + '-front', [x, y, z + c], X, Y, color),
    rectangle(id + '-back', [x, y, z - c], X, Y, color),
  ];
}

// The four pieces meet at the window edges. The opening has no surface.
function windowWall(id, y0, y1, z0, z1) {
  return rectangle(id, [-3.4, (y0 + y1) / 2, (z0 + z1) / 2],
    [0, 0, (z1 - z0) / 2], [0, (y1 - y0) / 2, 0], plaster);
}

const limestoneFloor = rectangle('floor', [0, 0, 0.05], [3.4, 0, 0], [0, 0, 4.45], limestone);
const walls = [
  rectangle('ceiling', [0, 3.7, 0.05], [3.4, 0, 0], [0, 0, 4.45], plaster),
  rectangle('north-wall', [0, 1.85, -4.4], [3.4, 0, 0], [0, 1.85, 0], plaster),
  rectangle('south-wall', [0, 1.85, 4.5], [3.4, 0, 0], [0, 1.85, 0], plaster),
  rectangle('east-wall', [3.4, 1.85, 0.05], [0, 0, 4.45], [0, 1.85, 0], [0.69, 0.67, 0.55]),
  windowWall('window-below', 0, 0.94, -4.4, 4.5),
  windowWall('window-above', 3.10, 3.7, -4.4, 4.5),
  windowWall('window-rear', 0.94, 3.10, -4.4, -3.25),
  windowWall('window-front', 0.94, 3.10, 0.75, 4.5),
  rectangle('exterior-wall', [-6.9, 2.5, -1.5], [0, 0, 15], [0, 7, 0], [0.58, 0.69, 0.72]),
];

const floors = {
  limestone: limestoneFloor,
  red: { ...limestoneFloor, color: [0.73, 0.055, 0.036] },
  blue: { ...limestoneFloor, color: [0.046, 0.12, 0.74] },
};

function furniture(alternate) {
  const x = alternate ? 1.05 : -0.40;
  const z = alternate ? -0.68 : -1.62;
  const pictureX = alternate ? -0.95 : 0.25;
  return [
    ...block('bench-seat', [x, 0.56, z], [2.20, 0.12, 0.65], oak),
    ...block('bench-left-leg', [x - 0.85, 0.25, z], [0.13, 0.50, 0.53], charcoal, false),
    ...block('bench-right-leg', [x + 0.85, 0.25, z], [0.13, 0.50, 0.53], charcoal, false),
    rectangle('print-paper', [pictureX, 1.94, -4.36], [0.86, 0, 0], [0, 0.98, 0], chalk),
    rectangle('print-clay', [pictureX - 0.03, 1.91, -4.354], [0.68, 0, 0], [0, 0.70, 0], clay),
    rectangle('print-ink', [pictureX + 0.11, 1.69, -4.348], [0.29, 0, 0], [0, 0.38, 0], blue),
  ];
}

const furnishings = { studio: furniture(false), alternate: furniture(true) };

function blind(id, x) {
  return function tilt(angle, radius, count) {
    const c=Math.cos(angle),s=Math.sin(angle),axes=[[c,-s,0],[s,c,0],[0,0,1]];
    return Array.from({length:count},(_,i)=>roundedBox(id+'-'+i,
      [x,(202+24*i-12*(count-1))+'/100','-1.25'],axes,
      count===1?['0.17','0.045','1.75']:['0.113','0.012','2'],radius,oak));
  };

}

const windowBlind = blind('window-slat', '-3.4');
const interiorBlind = blind('interior-slat', '-3.03');

export function buildRoom({
  angle = 24 * Math.PI / 180,
  secondBlind = false,
  secondAngle = -17 * Math.PI / 180,
  slats = 9,
  radius = slats===1 ? "0.028" : "0.008",
  secondRadius = radius,
  layout = 'studio',
  floor = 'limestone',
} = {}) {
  return {
    rectangles: [
      floors[floor],
      ...walls,
      ...furnishings[layout],
    ],
    roundedBoxes: [
      ...windowBlind(angle,radius,slats),
      ...(secondBlind ? interiorBlind(secondAngle,secondRadius,slats) : []),
    ],
    lightPosition,
    lampRGB: DEFAULT_LAMP_RGB,
  };
}
