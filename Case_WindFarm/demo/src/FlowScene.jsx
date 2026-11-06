import React, { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { directoryView, fieldColor, fieldValue, getFieldRange, sampleVelocity, sliceSampleCoordinates, traceStreamlines, validGrid } from './flowMath.js';
import { buildOrganizationVenn } from './organizationVenn.js';

const clamp = (v, a, b) => Math.max(a, Math.min(b, v));

function clearGroup(group) {
  while (group.children.length) {
    const object = group.children[0];
    group.remove(object);
    object.traverse((part) => {
      part.geometry?.dispose();
      const materials = Array.isArray(part.material) ? part.material : [part.material];
      for (const material of materials) {
        if (!material) continue;
        material.map?.dispose();
        material.dispose();
      }
    });
  }
}

function sprite(text, scale = 1) {
  const canvas = document.createElement('canvas');
  canvas.width = 256;
  canvas.height = 80;
  const context = canvas.getContext('2d');
  context.font = '500 34px Inter, system-ui, sans-serif';
  context.fillStyle = '#b4cecd';
  context.textAlign = 'center';
  context.textBaseline = 'middle';
  context.fillText(text, 128, 40);
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  const material = new THREE.SpriteMaterial({ map: texture, transparent: true, depthWrite: false });
  const label = new THREE.Sprite(material);
  label.scale.set(3.6 * scale, 1.125 * scale, 1);
  return label;
}

function segments(points, color, opacity = 1) {
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.Float32BufferAttribute(points, 3));
  return new THREE.LineSegments(geometry, new THREE.LineBasicMaterial({ color, transparent: opacity < 1, opacity, depthWrite: false }));
}

function lineLoop(points, color, opacity) {
  const geometry = new THREE.BufferGeometry().setFromPoints(points.map((p) => new THREE.Vector3(...p)));
  return new THREE.LineLoop(geometry, new THREE.LineBasicMaterial({ color, transparent: true, opacity, depthWrite: false }));
}

function toWorld(engine, point) {
  return [point[0] - engine.cx, point[2] * engine.exaggeration, point[1] - engine.cy];
}

function addGround(engine, domain) {
  const group = engine.ground;
  const { min, max } = domain;
  const dx = max[0] - min[0];
  const dy = max[1] - min[1];
  const ground = new THREE.Mesh(
    new THREE.PlaneGeometry(dx, dy),
    new THREE.MeshBasicMaterial({ color: 0x0c252b, transparent: true, opacity: 0.38, side: THREE.DoubleSide, depthWrite: false }),
  );
  ground.rotation.x = -Math.PI / 2;
  ground.position.y = -0.025;
  group.add(ground);
  const points = [];
  const targetSpacing = Math.max(dx, dy) / 16;
  const power = 10 ** Math.floor(Math.log10(targetSpacing));
  const scaled = targetSpacing / power;
  const spacing = (scaled <= 1 ? 1 : scaled <= 2 ? 2 : scaled <= 5 ? 5 : 10) * power;
  for (let x = Math.ceil(min[0] / spacing) * spacing; x <= max[0] + 1e-6; x += spacing) {
    points.push(...toWorld(engine, [x, min[1], 0]), ...toWorld(engine, [x, max[1], 0]));
  }
  for (let y = Math.ceil(min[1] / spacing) * spacing; y <= max[1] + 1e-6; y += spacing) {
    points.push(...toWorld(engine, [min[0], y, 0]), ...toWorld(engine, [max[0], y, 0]));
  }
  group.add(segments(points, 0x279690, 0.3));
  const corners = [[min[0], min[1], 0], [max[0], min[1], 0], [max[0], max[1], 0], [min[0], max[1], 0]].map((p) => toWorld(engine, p));
  group.add(lineLoop(corners, 0x75b8ad, 0.55));
  const axisOrigin = [min[0], min[1], 0];
  const verticalGuide = Math.min(max[2], 1.5);
  group.add(segments([
    ...toWorld(engine, axisOrigin), ...toWorld(engine, [max[0], min[1], 0]),
    ...toWorld(engine, axisOrigin), ...toWorld(engine, [min[0], max[1], 0]),
    ...toWorld(engine, axisOrigin), ...toWorld(engine, [min[0], min[1], verticalGuide]),
  ], 0xa9d0cd, 0.6));
  const labelScale = Math.max(dx, dy) / 38;
  const labelSpecs = [
    ['X (D)', [(min[0] + max[0]) / 2, min[1] - spacing * 0.75, 0]],
    ['Y (D)', [min[0] - spacing * 0.8, (min[1] + max[1]) / 2, 0]],
    ['Z (D)', [min[0] - spacing * 0.55, min[1], verticalGuide * 0.6]],
  ];
  for (const [text, position] of labelSpecs) {
    const label = sprite(text, labelScale);
    label.position.set(...toWorld(engine, position));
    group.add(label);
  }
}

function makeTurbine(engine, turbine) {
  const wrapper = new THREE.Group();
  wrapper.userData.turbineId = turbine.id;
  wrapper.position.set(turbine.x - engine.cx, 0, turbine.y - engine.cy);
  const actual = new THREE.Group();
  actual.scale.y = engine.exaggeration;
  wrapper.add(actual);

  const hubHeight = Number.isFinite(turbine.z) ? turbine.z : 0.875;
  const radius = Number.isFinite(turbine.rotor_diameter) ? turbine.rotor_diameter / 2 : 0.5;
  const ivory = new THREE.MeshStandardMaterial({ color: 0xe2e9e5, roughness: 0.62, metalness: 0.05 });
  const tower = new THREE.Mesh(new THREE.CylinderGeometry(0.038, 0.065, hubHeight, 12), ivory);
  tower.position.y = hubHeight / 2;
  actual.add(tower);

  const head = new THREE.Group();
  head.position.y = hubHeight;
  // Native x/y are already downstream/crosswind for every direction category;
  // the case's rotor disk is in its Y-Z plane, without a second rotation.
  head.rotation.y = Math.PI / 2;
  actual.add(head);
  const nacelle = new THREE.Mesh(new THREE.CylinderGeometry(0.045, 0.043, 0.17, 12), ivory.clone());
  nacelle.rotation.x = Math.PI / 2;
  nacelle.position.z = -0.045;
  head.add(nacelle);
  const hub = new THREE.Mesh(new THREE.SphereGeometry(0.052, 14, 10), ivory.clone());
  hub.scale.set(1, 1, 1.4);
  hub.position.z = 0.065;
  head.add(hub);
  const rotor = new THREE.Group();
  rotor.position.z = 0.09;
  head.add(rotor);
  const blade = new THREE.Shape();
  blade.moveTo(-0.021, 0.025);
  blade.bezierCurveTo(-0.038, radius * 0.27, -0.026, radius * 0.70, -0.004, radius);
  blade.bezierCurveTo(0.004, radius * 1.01, 0.025, radius * 0.78, 0.040, radius * 0.48);
  blade.bezierCurveTo(0.050, radius * 0.24, 0.035, 0.055, 0.021, 0.025);
  blade.closePath();
  for (let i = 0; i < 3; i += 1) {
    const mesh = new THREE.Mesh(new THREE.ShapeGeometry(blade, 10), new THREE.MeshBasicMaterial({ color: 0xf3fff8, side: THREE.DoubleSide }));
    mesh.rotation.z = i * Math.PI * 2 / 3;
    rotor.add(mesh);
  }
  engine.rotors.push(rotor);

  const baseRing = new THREE.Mesh(new THREE.RingGeometry(0.10, 0.125, 28), new THREE.MeshBasicMaterial({ color: 0x64ddce, transparent: true, opacity: 0.55, side: THREE.DoubleSide, depthWrite: false }));
  baseRing.rotation.x = -Math.PI / 2;
  baseRing.position.y = 0.015;
  wrapper.add(baseRing);
  wrapper.userData.ring = baseRing;
  // A slightly larger invisible target makes real rotor-sized turbines selectable.
  const hit = new THREE.Mesh(new THREE.CylinderGeometry(Math.max(0.22, radius * 0.65), Math.max(0.22, radius * 0.65), hubHeight + radius * 0.6, 8), new THREE.MeshBasicMaterial({ colorWrite: false, depthWrite: false }));
  hit.position.y = (hubHeight + radius * 0.6) / 2;
  actual.add(hit);
  wrapper.traverse((object) => { object.userData.turbineId = turbine.id; });
  return wrapper;
}

function updateSelection(engine, selected) {
  for (const turbine of engine.turbines.children) {
    const ring = turbine.userData.ring;
    const active = turbine.userData.turbineId === selected;
    ring.material.color.set(active ? 0x80ffe4 : 0x64ddce);
    ring.material.opacity = active ? 1 : 0.5;
    ring.scale.setScalar(active ? 1.7 : 1);
  }
}

function cameraView(engine, preset = 'perspective') {
  if (!engine.domain) return;
  if (engine.abstractActive && engine.orgBounds?.length) {
    fitWorldPoints(engine, engine.orgBounds, ['perspective', 'fit', 'top'].includes(preset) ? 'venn' : preset);
    return;
  }
  const { min, max } = engine.domain;
  const dx = max[0] - min[0];
  const dy = max[1] - min[1];
  const domainSize = Math.max(dx, dy);
  let size = domainSize;
  let targetX = 0;
  let targetZ = 0;
  if (preset === 'perspective' && engine.turbines.children.length) {
    const xs = engine.turbines.children.map((turbine) => turbine.position.x);
    const zs = engine.turbines.children.map((turbine) => turbine.position.z);
    targetX = (Math.min(...xs) + Math.max(...xs)) / 2;
    targetZ = (Math.min(...zs) + Math.max(...zs)) / 2;
    const layoutSpan = Math.max(Math.max(...xs) - Math.min(...xs), Math.max(...zs) - Math.min(...zs));
    size = clamp(layoutSpan + 7, domainSize * 0.48, domainSize * 0.8);
  }
  const target = new THREE.Vector3(targetX, 0.8 * engine.exaggeration, targetZ);
  engine.controls.target.copy(target);
  const aspectFit = Math.max(1, 0.9 / engine.camera.aspect);
  if (preset === 'top') engine.camera.position.set(targetX + 0.001, size * 1.35 * aspectFit, targetZ + 0.001);
  else if (preset === 'front') engine.camera.position.set(targetX, size * 0.23 * aspectFit, targetZ + size * 1.35 * aspectFit);
  else if (preset === 'side') engine.camera.position.set(targetX + size * 1.35 * aspectFit, size * 0.23 * aspectFit, targetZ);
  else engine.camera.position.set(targetX + size * 0.9 * aspectFit, size * 0.68 * aspectFit, targetZ + size * 1.04 * aspectFit);
  engine.camera.near = Math.max(size / 5000, 0.001);
  engine.camera.far = size * 12;
  engine.camera.updateProjectionMatrix();
  engine.controls.minDistance = size * 0.08;
  engine.controls.maxDistance = size * 4;
  engine.controls.update();
}

function rebuildScene(engine, scene, view, selected) {
  if (!scene?.domain?.min || !scene?.domain?.max) return;
  clearGroup(engine.ground);
  clearGroup(engine.turbines);
  engine.rotors = [];
  engine.domain = scene.domain;
  engine.exaggeration = clamp(Number(view.exaggeration) || 1, 0.25, 10);
  engine.cx = (scene.domain.min[0] + scene.domain.max[0]) / 2;
  engine.cy = (scene.domain.min[1] + scene.domain.max[1]) / 2;
  addGround(engine, scene.domain);
  for (const turbine of scene.turbines || []) {
    if (Number.isFinite(turbine.x) && Number.isFinite(turbine.y)) engine.turbines.add(makeTurbine(engine, turbine));
  }
  updateSelection(engine, selected);
  const signature = [scene.id, ...scene.domain.min, ...scene.domain.max].join(':');
  if (engine.sceneSignature !== signature) {
    engine.sceneSignature = signature;
    engine.orgAutoFitted = false;
    cameraView(engine, engine.cameraPreset);
  }
}

function vertexColor(value, range, field) {
  const rgb = fieldColor(value, range, field);
  const c = new THREE.Color().setRGB(...rgb, THREE.SRGBColorSpace);
  return [c.r, c.g, c.b];
}

function rebuildSlice(engine) {
  clearGroup(engine.slice);
  const { result, view } = engine;
  const grid = result?.grid;
  if (!view.slice || !validGrid(grid) || !engine.domain) return;
  const range = engine.range;
  if (!range) return;
  const deficitRange = getFieldRange(grid, 'deficit', { contrast: view.contrast, turbines: engine.scene?.turbines });
  const deficitScale = Math.max(0.25, deficitRange?.max || 0);
  const sampling = sliceSampleCoordinates(grid, view.sliceAxis, view.slicePosition);
  const [countA, countB] = sampling.shape;
  const vertices = [];
  const colors = [];
  const valid = [];
  const indices = [];
  for (let j = 0; j < countB; j += 1) for (let i = 0; i < countA; i += 1) {
    const p = sampling.coordinates[j * countA + i];
    const v = sampleVelocity(grid, p);
    vertices.push(...toWorld(engine, p));
    const value = v && fieldValue(v, view.field, p, grid);
    const deficit = v && fieldValue(v, 'deficit', p, grid);
    const isValid = v && Number.isFinite(value);
    valid.push(!!isValid);
    // Opacity shows the actual positive profile-relative deficit; the color
    // remains the selected physical field on its shared, labeled scale.
    const strength = Number.isFinite(deficit) ? clamp(deficit / deficitScale, 0, 1) : 0;
    const alpha = 0.08 + 0.54 * strength * strength * (3 - 2 * strength);
    colors.push(...(isValid ? [...vertexColor(value, range, view.field), alpha] : [0, 0, 0, 0]));
  }
  for (let j = 0; j < countB - 1; j += 1) for (let i = 0; i < countA - 1; i += 1) {
    const k = j * countA + i;
    if (valid[k] && valid[k + 1] && valid[k + countA]) indices.push(k, k + 1, k + countA);
    if (valid[k + 1] && valid[k + countA + 1] && valid[k + countA]) indices.push(k + 1, k + countA + 1, k + countA);
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.Float32BufferAttribute(vertices, 3));
  geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 4));
  geometry.setIndex(indices);
  geometry.computeVertexNormals();
  const material = new THREE.MeshBasicMaterial({ vertexColors: true, transparent: true, opacity: 1, side: THREE.DoubleSide, depthWrite: false });
  const mesh = new THREE.Mesh(geometry, material);
  mesh.renderOrder = 2;
  engine.slice.add(mesh);
  const edge = [0, countA - 1, countA * countB - 1, countA * (countB - 1)].map((index) => toWorld(engine, sampling.coordinates[index]));
  engine.slice.add(lineLoop(edge, 0x67f1dc, 0.22));
}

const ADDRESS_COLORS = [0x78f5d0, 0x65d5f4, 0xa5efdc, 0x40d7b8, 0x7bb9f5, 0x83e2ca, 0x4fb7ce, 0xb6f6e9, 0x55e5d6, 0x90c7f0, 0xb6e4c6, 0x6ac4c2];

function organizationMode(engine) {
  return engine.view.mode === 'organization';
}

function applyMode(engine) {
  const org = organizationMode(engine);
  const abstract = org && engine.view.organizationView === 'venn';
  if (abstract && !engine.abstractActive) {
    engine.physicalCamera = { position: engine.camera.position.clone(), target: engine.controls.target.clone(), near: engine.camera.near, far: engine.camera.far, minDistance: engine.controls.minDistance, maxDistance: engine.controls.maxDistance };
    engine.abstractActive = true;
    engine.orgAutoFitted = false;
  } else if (!abstract && engine.abstractActive) {
    const saved = engine.physicalCamera;
    engine.abstractActive = false;
    engine.orgAutoFitted = false;
    engine.orgBounds = null;
    if (saved) {
      engine.camera.position.copy(saved.position);
      engine.controls.target.copy(saved.target);
      engine.camera.near = saved.near;
      engine.camera.far = saved.far;
      engine.camera.updateProjectionMatrix();
      engine.controls.minDistance = saved.minDistance;
      engine.controls.maxDistance = saved.maxDistance;
      engine.controls.update();
    }
    engine.physicalCamera = null;
  }
  engine.ground.visible = !abstract;
  engine.turbines.visible = !abstract;
  engine.flow.visible = !org;
  engine.slice.visible = !org;
  // Keep the prior overlay behavior for callers without a mode selector.
  engine.organization.visible = org || (!engine.view.mode && !!engine.view.organization);
}

function fitWorldPoints(engine, coordinates, preset = null) {
  const points = coordinates.map((point) => new THREE.Vector3(...point));
  if (!points.length) return;
  const bounds = new THREE.Box3().setFromPoints(points);
  const center = bounds.getCenter(new THREE.Vector3());
  const presets = { top: [0.001, 1, 0.001], front: [0, 0.23, 1.35], side: [1.35, 0.23, 0], perspective: [0.9, 0.68, 1.04], venn: [0, 0.0151, 1] };
  const direction = preset && presets[preset] ? new THREE.Vector3(...presets[preset]).normalize() : engine.camera.position.clone().sub(engine.controls.target).normalize();
  const right = new THREE.Vector3().crossVectors(engine.camera.up, direction).normalize();
  const up = new THREE.Vector3().crossVectors(direction, right).normalize();
  const tanVertical = Math.tan(engine.camera.fov * Math.PI / 360);
  const tanHorizontal = tanVertical * engine.camera.aspect;
  let distance = 0;
  for (const point of points) {
    const delta = point.clone().sub(center);
    const depth = delta.dot(direction);
    distance = Math.max(distance, depth + Math.abs(delta.dot(right)) / tanHorizontal, depth + Math.abs(delta.dot(up)) / tanVertical);
  }
  distance = Math.max(distance * 1.16, engine.controls.minDistance);
  engine.controls.target.copy(center);
  engine.camera.position.copy(center).addScaledVector(direction, distance);
  engine.camera.near = Math.max(distance / 5000, 0.001);
  engine.camera.far = Math.max(distance * 12, 100);
  engine.camera.updateProjectionMatrix();
  engine.controls.minDistance = Math.min(engine.controls.minDistance, distance * 0.08);
  engine.controls.maxDistance = Math.max(engine.controls.maxDistance, distance * 1.5);
  engine.controls.update();
}

function fitOrganization(engine, scene, data) {
  if (engine.orgAutoFitted || !organizationMode(engine) || !data.anchors.length) return;
  const points = data.anchors.map((anchor) => toWorld(engine, anchor.position));
  for (const turbine of scene?.turbines || []) {
    if ([turbine.x, turbine.y, turbine.z].every(Number.isFinite)) points.push(toWorld(engine, [turbine.x, turbine.y, turbine.z]));
  }
  fitWorldPoints(engine, points);
  engine.orgAutoFitted = true;
}

function pointsObject(positions, materialOptions) {
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
  return new THREE.Points(geometry, new THREE.PointsMaterial({
    sizeAttenuation: false, transparent: true, depthWrite: false, ...materialOptions,
  }));
}

function glowTexture() {
  const canvas = document.createElement('canvas');
  canvas.width = 64;
  canvas.height = 64;
  const context = canvas.getContext('2d');
  const gradient = context.createRadialGradient(32, 32, 0, 32, 32, 32);
  gradient.addColorStop(0, 'rgba(255,255,255,1)');
  gradient.addColorStop(0.22, 'rgba(255,255,255,0.75)');
  gradient.addColorStop(1, 'rgba(255,255,255,0)');
  context.fillStyle = gradient;
  context.fillRect(0, 0, 64, 64);
  return new THREE.CanvasTexture(canvas);
}

function memberLabel(text, color, scale) {
  const canvas = document.createElement('canvas');
  canvas.width = 192;
  canvas.height = 80;
  const context = canvas.getContext('2d');
  context.font = '600 41px Inter, system-ui, sans-serif';
  context.textAlign = 'center';
  context.textBaseline = 'middle';
  context.lineWidth = 6;
  context.strokeStyle = '#071720';
  context.strokeText(text, 96, 40);
  context.fillStyle = color;
  context.fillText(text, 96, 40);
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  const label = new THREE.Sprite(new THREE.SpriteMaterial({ map: texture, transparent: true, depthWrite: false, depthTest: false }));
  label.scale.set(3.2 * scale, 1.33 * scale, 1);
  label.renderOrder = 4;
  return label;
}

function curveAt(path, t) {
  const a = (1 - t) * (1 - t);
  const b = 2 * t * (1 - t);
  const c = t * t;
  return [0, 1, 2].map((i) => path.start[i] * a + path.control[i] * b + path.end[i] * c);
}

function rebuildOrganization(engine, scene) {
  clearGroup(engine.organization);
  engine.orgPulses = null;
  const organization = engine.result?.organization;
  if (organizationMode(engine) && engine.view.organizationView === 'venn') {
    const venn = buildOrganizationVenn({ THREE, organization, scene, selectedAddress: engine.view.organizationAddress, toWorld: (point) => toWorld(engine, point) });
    applyMode(engine);
    if (venn) {
      engine.organization.add(venn.group);
      engine.orgBounds = venn.anchors;
      engine.onOrganizationSummary?.(venn.summary);
      if (!engine.orgAutoFitted) {
        fitWorldPoints(engine, venn.anchors, 'venn');
        engine.orgAutoFitted = true;
      }
    } else {
      engine.orgBounds = null;
      engine.onOrganizationSummary?.(null);
    }
    return;
  }
  const data = directoryView(organization, scene, engine.view.organizationAddress);
  if (!engine.domain || !data) {
    engine.onOrganizationSummary?.(null);
    applyMode(engine);
    return;
  }
  const { anchors, rows, links, missing } = data;
  const fullRowIds = organization.selection?.selected_anchor_ids || rows.map((row) => row.id);
  const palette = new Map(fullRowIds.map((id, i) => [String(id), ADDRESS_COLORS[i % ADDRESS_COLORS.length]]));
  const selectedRowIds = new Set(rows.map((row) => String(row.id)));
  const selfIds = new Set(links.filter((link) => link.isSelf).map((link) => String(link.receiverId)));
  const positions = [];
  const colors = [];
  const paths = [];
  const glow = glowTexture();
  const extent = Math.max(engine.domain.max[0] - engine.domain.min[0], engine.domain.max[1] - engine.domain.min[1]);
  const arrowLength = clamp(extent * 0.009, 0.14, 0.35);
  const bundleIndex = new Map(rows.map((row, index) => [String(row.id), index]));
  const addSegment = (a, b, color) => {
    positions.push(...a, ...b);
    colors.push(color.r, color.g, color.b, color.r, color.g, color.b);
  };
  for (const link of links) {
    // Own-E membership is rendered as a ring around its actual receiver vertex.
    if (link.isSelf) continue;
    const start = toWorld(engine, link.donorPosition);
    const end = toWorld(engine, link.receiverPosition);
    const distance = Math.hypot(end[0] - start[0], end[1] - start[1], end[2] - start[2]);
    const lift = Math.min(extent * 0.13, 0.2 + distance * 0.13) * (1 + 0.04 * (bundleIndex.get(String(link.receiverId)) || 0));
    const control = [(start[0] + end[0]) / 2, (start[1] + end[1]) / 2 + lift, (start[2] + end[2]) / 2];
    const path = { start, control, end };
    const color = new THREE.Color(palette.get(String(link.receiverId)) || ADDRESS_COLORS[0]);
    // Fixed logarithmic brightness scale 1e-6 to 1; smaller positive weights
    // remain visible at the floor and are not discarded.
    const strength = 0.20 + 0.80 * clamp((Math.log10(link.weight) + 6) / 6, 0, 1);
    color.multiplyScalar(strength);
    for (let step = 0; step < 8; step += 1) addSegment(curveAt(path, step / 8), curveAt(path, (step + 1) / 8), color);
    const tip = new THREE.Vector3(...curveAt(path, 0.82));
    const before = new THREE.Vector3(...curveAt(path, 0.78));
    const direction = tip.clone().sub(before).normalize();
    let side = new THREE.Vector3(-direction.z, 0, direction.x);
    if (side.lengthSq() < 1e-6) side = new THREE.Vector3(1, 0, 0);
    side.normalize().multiplyScalar(arrowLength * 0.42);
    const tail = tip.clone().addScaledVector(direction, -arrowLength);
    addSegment(tip.toArray(), tail.clone().add(side).toArray(), color);
    addSegment(tip.toArray(), tail.clone().sub(side).toArray(), color);
    paths.push({ ...path, color: [color.r, color.g, color.b] });
  }
  if (positions.length) {
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
    geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
    const lines = new THREE.LineSegments(geometry, new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, opacity: 0.58, depthWrite: false }));
    engine.organization.add(lines);
    const pulsePositions = [];
    const pulseColors = [];
    paths.forEach((path, i) => {
      pulsePositions.push(...curveAt(path, (i * 0.173) % 1));
      pulseColors.push(...path.color);
    });
    const pulses = pointsObject(pulsePositions, { size: 6, opacity: 0.9, map: glow, vertexColors: true, blending: THREE.AdditiveBlending });
    pulses.geometry.getAttribute('position').setUsage(THREE.DynamicDrawUsage);
    pulses.geometry.setAttribute('color', new THREE.Float32BufferAttribute(pulseColors, 3));
    pulses.renderOrder = 3;
    engine.organization.add(pulses);
    engine.orgPulses = { paths, geometry: pulses.geometry };
  }
  const allE = anchors.flatMap((anchor) => toWorld(engine, anchor.position));
  const receiverE = rows.flatMap((row) => toWorld(engine, row.position));
  const ownE = anchors.filter((anchor) => selfIds.has(String(anchor.id))).flatMap((anchor) => toWorld(engine, anchor.position));
  const allM = (scene?.turbines || []).filter((member) => [member.x, member.y, member.z].every(Number.isFinite))
    .flatMap((member) => toWorld(engine, [member.x, member.y, member.z]));
  if (allE.length) {
    engine.organization.add(pointsObject(allE, { color: 0x6bb9c3, size: 15, opacity: 0.33, map: glow, blending: THREE.AdditiveBlending }));
    engine.organization.add(pointsObject(allE, { color: 0x9ce7e0, size: 3, opacity: 0.9 }));
  }
  if (receiverE.length) {
    engine.organization.add(pointsObject(receiverE, { color: 0x81ffe1, size: 30, opacity: 0.48, map: glow, blending: THREE.AdditiveBlending }));
    engine.organization.add(pointsObject(receiverE, { color: 0xe1fff8, size: 7, opacity: 1 }));
  }
  if (ownE.length) {
    // A thin circular marker identifies positive own-E membership, without an invented edge.
    for (const anchor of anchors.filter((item) => selfIds.has(String(item.id)))) {
      const marker = new THREE.Mesh(new THREE.RingGeometry(0.14, 0.19, 24), new THREE.MeshBasicMaterial({
        color: palette.get(String(anchor.id)) || ADDRESS_COLORS[0], transparent: true, opacity: 0.9, side: THREE.DoubleSide, depthWrite: false,
      }));
      marker.rotation.x = -Math.PI / 2;
      marker.position.set(...toWorld(engine, anchor.position));
      engine.organization.add(marker);
    }
  }
  if (allM.length) {
    engine.organization.add(pointsObject(allM, { color: 0x75ffe0, size: 22, opacity: 0.45, map: glow, blending: THREE.AdditiveBlending }));
    engine.organization.add(pointsObject(allM, { color: 0xe5fff6, size: 8, opacity: 1 }));
  }
  const labelScale = clamp(extent / 75, 0.45, 0.8);
  for (const member of scene?.turbines || []) {
    if (![member.x, member.y, member.z].every(Number.isFinite)) continue;
    const label = memberLabel(`M${member.id}`, '#aaffdf', labelScale);
    label.position.set(...toWorld(engine, [member.x, member.y, member.z + 0.46]));
    engine.organization.add(label);
  }
  for (const anchor of anchors) {
    if (!selectedRowIds.has(String(anchor.id))) continue;
    const label = memberLabel(`E${anchor.id}`, '#8edcf9', labelScale);
    label.position.set(...toWorld(engine, [anchor.position[0], anchor.position[1], anchor.position[2] + 0.4]));
    engine.organization.add(label);
  }
  engine.onOrganizationSummary?.({
    shownRows: rows.length,
    totalRows: organization.selection?.total_rows ?? anchors.length,
    rowIds: rows.map((row) => row.id),
    positiveMembers: links.length + missing,
    drawnMembers: links.length,
    selfMembers: selfIds.size,
    missingMembers: missing,
  });
  applyMode(engine);
  fitOrganization(engine, scene, data);
}

function updateOrganizationPulses(engine, time) {
  if (!engine.orgPulses || !engine.organization.visible) return;
  const { paths, geometry } = engine.orgPulses;
  const positions = geometry.getAttribute('position');
  paths.forEach((path, i) => positions.setXYZ(i, ...curveAt(path, (time * 0.14 + i * 0.173) % 1)));
  positions.needsUpdate = true;
}

function rebuildLines(engine) {
  clearGroup(engine.flow);
  engine.moving = null;
  engine.movingPaths = [];
  const { result, view, paths } = engine;
  if (!view.streamlines || !validGrid(result?.grid) || !engine.domain || !paths.length) return;
  const range = engine.range;
  if (!range) return;
  const linesByKind = { background: { positions: [], colors: [] }, rotor: { positions: [], colors: [] } };
  const movingPositions = [];
  const movingColors = [];
  for (const path of paths) {
    const kind = path.kind === 'background' ? 'background' : 'rotor';
    const { positions, colors } = linesByKind[kind];
    const selectedValues = path.map((item) => fieldValue(item.velocity, view.field, item.position, result.grid));
    if (!selectedValues.every(Number.isFinite)) continue;
    for (let i = 1; i < path.length; i += 1) {
      for (const index of [i - 1, i]) {
        const item = path[index];
        positions.push(...toWorld(engine, item.position));
        colors.push(...vertexColor(selectedValues[index], range, view.field));
      }
    }
    for (let streak = 0; streak < 2; streak += 1) {
      const sample = path[Math.floor(path.length * (streak + 1) / 3)];
      movingPositions.push(...toWorld(engine, sample.position), ...toWorld(engine, sample.position));
      const color = vertexColor(fieldValue(sample.velocity, view.field, sample.position, result.grid), range, view.field);
      movingColors.push(...color, ...color);
    }
    engine.movingPaths.push(path);
  }
  for (const [kind, { positions, colors }] of Object.entries(linesByKind)) {
    if (!positions.length) continue;
    const lineGeometry = new THREE.BufferGeometry();
    lineGeometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
    lineGeometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
    const lines = new THREE.LineSegments(lineGeometry, new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, opacity: kind === 'background' ? 0.15 : 0.44, depthWrite: false }));
    engine.flow.add(lines);
  }
  if (!movingPositions.length) return;
  const streakGeometry = new THREE.BufferGeometry();
  streakGeometry.setAttribute('position', new THREE.Float32BufferAttribute(movingPositions, 3).setUsage(THREE.DynamicDrawUsage));
  streakGeometry.setAttribute('color', new THREE.Float32BufferAttribute(movingColors, 3).setUsage(THREE.DynamicDrawUsage));
  const streaks = new THREE.LineSegments(streakGeometry, new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, opacity: 0.82, depthWrite: false }));
  streaks.renderOrder = 3;
  engine.flow.add(streaks);
  engine.moving = streakGeometry;
}

function updateStreaks(engine, time) {
  if (!engine.moving) return;
  const positions = engine.moving.getAttribute('position');
  const colors = engine.moving.getAttribute('color');
  const range = engine.range;
  let vertex = 0;
  engine.movingPaths.forEach((path, pathIndex) => {
    for (let streak = 0; streak < 2; streak += 1) {
      const offset = (time * 0.085 + pathIndex * 0.137 + streak * 0.5) % 1;
      const end = Math.max(1, Math.min(path.length - 1, Math.floor(offset * (path.length - 1))));
      const start = Math.max(0, end - 3);
      for (const index of [start, end]) {
        const point = toWorld(engine, path[index].position);
        positions.setXYZ(vertex, ...point);
        if (range) colors.setXYZ(vertex, ...vertexColor(fieldValue(path[index].velocity, engine.view.field, path[index].position, engine.result.grid), range, engine.view.field));
        vertex += 1;
      }
    }
  });
  positions.needsUpdate = true;
  colors.needsUpdate = true;
}

/** Render only supplied model predictions; coordinates are rotor diameters. */
export default function FlowScene({ scene, result, view = {}, fieldRange = null, selected = null, onSelect, onDraggingChange, onTurbineMove, onOrganizationSummary, cameraPreset = 'perspective', cameraRevision = 0, onReady }) {
  const hostRef = useRef(null);
  const engineRef = useRef(null);
  const callbacksRef = useRef({ onSelect, onDraggingChange, onTurbineMove, onOrganizationSummary, onReady });
  callbacksRef.current = { onSelect, onDraggingChange, onTurbineMove, onOrganizationSummary, onReady };
  const [error, setError] = useState('');

  useEffect(() => {
    const host = hostRef.current;
    let renderer;
    try {
      renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false, powerPreference: 'high-performance' });
    } catch (cause) {
      setError('The 3D view could not start. Enable WebGL or open this page in a browser with graphics acceleration.');
      return undefined;
    }
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 1.5));
    renderer.setClearColor(0x071720, 1);
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.domElement.style.cssText = 'display:block;width:100%;height:100%;touch-action:none;outline:none;';
    renderer.domElement.setAttribute('aria-label', 'Interactive three-dimensional wind farm and reconstructed flow');
    host.appendChild(renderer.domElement);
    const world = new THREE.Scene();
    world.fog = new THREE.FogExp2(0x071720, 0.004);
    const camera = new THREE.PerspectiveCamera(38, 1, 0.01, 1000);
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.07;
    controls.screenSpacePanning = true;
    controls.maxPolarAngle = Math.PI / 2 - 0.015;
    world.add(new THREE.HemisphereLight(0xd7fff6, 0x24404c, 2.0));
    const light = new THREE.DirectionalLight(0xf4fff2, 2.3);
    light.position.set(-10, 25, 15);
    world.add(light);
    const engine = {
      renderer, world, camera, controls,
      ground: new THREE.Group(), turbines: new THREE.Group(), slice: new THREE.Group(), flow: new THREE.Group(), organization: new THREE.Group(),
      rotors: [], paths: [], view: {}, scene: null, domain: null, result: null, exaggeration: 1, cx: 0, cy: 0,
      cameraPreset: 'perspective', moving: null, orgPulses: null, range: null,
      onOrganizationSummary: (summary) => callbacksRef.current.onOrganizationSummary?.(summary),
    };
    world.add(engine.ground, engine.turbines, engine.flow, engine.slice, engine.organization);
    engineRef.current = engine;
    const resize = () => {
      const width = Math.max(1, host.clientWidth);
      const height = Math.max(1, host.clientHeight);
      renderer.setSize(width, height, false);
      camera.aspect = width / height;
      camera.updateProjectionMatrix();
    };
    const observer = new ResizeObserver(resize);
    observer.observe(host);
    resize();

    const raycaster = new THREE.Raycaster();
    const pointer = new THREE.Vector2();
    const dragPlane = new THREE.Plane(new THREE.Vector3(0, 1, 0), 0);
    const intersection = new THREE.Vector3();
    let drag = null;
    let down = null;
    const setPointer = (event) => {
      const rect = renderer.domElement.getBoundingClientRect();
      pointer.set((event.clientX - rect.left) / rect.width * 2 - 1, -(event.clientY - rect.top) / rect.height * 2 + 1);
      raycaster.setFromCamera(pointer, camera);
    };
    const pointerDown = (event) => {
      if (event.button !== 0 || engine.abstractActive) return;
      setPointer(event);
      const hit = raycaster.intersectObjects(engine.turbines.children, true).find((item) => item.object.userData.turbineId !== undefined);
      down = { x: event.clientX, y: event.clientY, id: hit?.object.userData.turbineId };
      if (!hit || engine.view.tool !== 'move') return;
      let turbine = hit.object;
      while (turbine.parent && turbine.parent !== engine.turbines) turbine = turbine.parent;
      if (turbine.parent !== engine.turbines || !raycaster.ray.intersectPlane(dragPlane, intersection)) return;
      drag = { id: turbine.userData.turbineId, turbine, x: turbine.position.x, y: turbine.position.z, dx: turbine.position.x - intersection.x, dy: turbine.position.z - intersection.z };
      engine.flow.visible = false;
      engine.slice.visible = false;
      engine.organization.visible = false;
      controls.enabled = false;
      renderer.domElement.setPointerCapture(event.pointerId);
      callbacksRef.current.onSelect?.(drag.id);
      callbacksRef.current.onDraggingChange?.(true);
      event.stopImmediatePropagation();
      event.preventDefault();
    };
    const pointerMove = (event) => {
      if (!drag || !engine.domain || engine.abstractActive) return;
      setPointer(event);
      if (raycaster.ray.intersectPlane(dragPlane, intersection)) {
        drag.turbine.position.x = clamp(intersection.x + drag.dx, engine.domain.min[0] - engine.cx, engine.domain.max[0] - engine.cx);
        drag.turbine.position.z = clamp(intersection.z + drag.dy, engine.domain.min[1] - engine.cy, engine.domain.max[1] - engine.cy);
      }
      event.preventDefault();
    };
    const pointerUp = (event) => {
      if (engine.abstractActive) {
        if (drag) callbacksRef.current.onDraggingChange?.(false);
        drag = null;
        down = null;
        controls.enabled = true;
        return;
      }
      if (drag) {
        const { id, turbine, x, y } = drag;
        controls.enabled = true;
        drag = null;
        if (event.type === 'pointercancel') {
          turbine.position.x = x;
          turbine.position.z = y;
        }
        if (Math.hypot(turbine.position.x - x, turbine.position.z - y) > 1e-5) {
          callbacksRef.current.onTurbineMove?.(id, turbine.position.x + engine.cx, turbine.position.z + engine.cy);
        } else {
          applyMode(engine);
        }
        callbacksRef.current.onDraggingChange?.(false);
        event.preventDefault();
      } else if (down && Math.hypot(event.clientX - down.x, event.clientY - down.y) < 4 && down.id !== undefined) {
        callbacksRef.current.onSelect?.(down.id);
      }
      down = null;
    };
    renderer.domElement.addEventListener('pointerdown', pointerDown, true);
    renderer.domElement.addEventListener('pointermove', pointerMove);
    renderer.domElement.addEventListener('pointerup', pointerUp);
    renderer.domElement.addEventListener('pointercancel', pointerUp);
    let frame;
    let previous = performance.now();
    let elapsed = 0;
    const animate = (now) => {
      const dt = Math.min((now - previous) / 1000, 0.05);
      previous = now;
      if (engine.view.playing !== false) {
        elapsed += dt;
        for (const rotor of engine.rotors) rotor.rotation.z -= dt * 0.55;
        updateStreaks(engine, elapsed);
        updateOrganizationPulses(engine, elapsed);
      }
      controls.update();
      renderer.render(world, camera);
      frame = requestAnimationFrame(animate);
    };
    frame = requestAnimationFrame(animate);
    callbacksRef.current.onReady?.();
    return () => {
      cancelAnimationFrame(frame);
      observer.disconnect();
      renderer.domElement.removeEventListener('pointerdown', pointerDown, true);
      renderer.domElement.removeEventListener('pointermove', pointerMove);
      renderer.domElement.removeEventListener('pointerup', pointerUp);
      renderer.domElement.removeEventListener('pointercancel', pointerUp);
      controls.dispose();
      for (const group of [engine.ground, engine.turbines, engine.slice, engine.flow, engine.organization]) clearGroup(group);
      renderer.dispose();
      renderer.domElement.remove();
      engineRef.current = null;
    };
  }, []);

  useEffect(() => {
    const engine = engineRef.current;
    if (!engine) return;
    engine.view = view;
    engine.scene = scene;
    rebuildScene(engine, scene, view, selected);
    rebuildSlice(engine);
    rebuildLines(engine);
    rebuildOrganization(engine, scene);
  }, [scene, view.exaggeration]);

  useEffect(() => {
    const engine = engineRef.current;
    if (!engine) return;
    engine.view = view;
    engine.result = result;
    engine.paths = validGrid(result?.grid) ? traceStreamlines(result.grid, scene?.turbines) : [];
    engine.range = fieldRange || getFieldRange(result?.grid, view.field, { contrast: view.contrast, turbines: scene?.turbines });
    engine.flow.visible = true;
    engine.slice.visible = true;
    rebuildSlice(engine);
    rebuildLines(engine);
    rebuildOrganization(engine, scene);
  }, [result]);

  useEffect(() => {
    const engine = engineRef.current;
    if (!engine) return;
    engine.view = view;
    engine.range = fieldRange || getFieldRange(engine.result?.grid, view.field, { contrast: view.contrast, turbines: scene?.turbines });
    rebuildSlice(engine);
    rebuildLines(engine);
  }, [view.field, view.contrast, view.slice, view.sliceAxis, view.slicePosition, view.streamlines, fieldRange]);

  useEffect(() => {
    const engine = engineRef.current;
    if (!engine) return;
    engine.view = view;
    engine.controls.mouseButtons.LEFT = view.tool === 'pan' ? THREE.MOUSE.PAN : THREE.MOUSE.ROTATE;
    applyMode(engine);
    rebuildOrganization(engine, scene);
  }, [view.playing, view.tool, view.organization, view.mode, view.organizationView]);

  useEffect(() => {
    const engine = engineRef.current;
    if (!engine) return;
    engine.view = view;
    rebuildOrganization(engine, scene);
  }, [view.organizationAddress]);

  useEffect(() => { if (engineRef.current) updateSelection(engineRef.current, selected); }, [selected]);
  useEffect(() => {
    const engine = engineRef.current;
    if (!engine) return;
    engine.cameraPreset = cameraPreset;
    cameraView(engine, cameraPreset);
  }, [cameraPreset, cameraRevision]);

  return <div className="flow-scene" ref={hostRef} style={{ width: '100%', height: '100%', minHeight: 0, position: 'relative' }}>
    {error && <div role="alert" style={{ position: 'absolute', inset: 0, display: 'grid', placeItems: 'center', padding: '2rem', background: '#071720', color: '#d7e8e8', textAlign: 'center', font: '14px/1.5 system-ui' }}>{error}</div>}
  </div>;
}
