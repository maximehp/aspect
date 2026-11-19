import { buildRoom } from './room.js';
import { roundedBox } from './rounded-box.js';

const axes = [
  [1, 0, 0],
  [0, 1, 0],
  [0, 0, 1],
];
// Replay game time one tick at a time, even if the machine misses a deadline.
// The camera travels with a shadow-casting body. A separate moving solid
// passes the bench while both sets of slats rotate independently.
export function motionPose(t) {
  const x = 0.9 + 0.65 * Math.sin(0.85 * t),
    z = 0.7 + 0.65 * Math.cos(1.15 * t);
  const scene = buildRoom({
    angle: 0.38 + 0.52 * Math.sin(2.1 * t),
    secondBlind: true,
    secondAngle: -0.25 + 0.44 * Math.sin(1.4 * t),
  });
  scene.roundedBoxes.push(
    roundedBox('player-body', [x, 0.75, z], axes, [0.21, 0.73, 0.16], 0.12, [0.18, 0.23, 0.39]),
    roundedBox(
      'moving-block',
      [-0.3 + 1.9 * Math.sin(1.35 * t), 0.37, -0.86],
      axes,
      [0.21, 0.36, 0.2],
      0.05,
      [0.6, 0.28, 0.1],
    ),
  );
  return {
    scene,
    camera: { position: [x, 1.62, z], yaw: -0.85 + 0.42 * Math.sin(1.1 * t), pitch: -0.13, fov: Math.PI / 3 },
  };
}
