# Traum Engine

A playful, *Dreams*-inspired 3D Signed Distance Field (SDF) & Painterly Fleck game creation engine built in **Rust** and **`wgpu`**.

Sculpt worlds out of warm, tactile SDF clay coated in living painterly brushstrokes ("flecks"), animate toy props live by waving them around, and jump straight into **Play Mode**—across **Mouse & Keyboard**, **Multi-Touch**, and **Game Controllers**.

Architecture & full multi-milestone roadmap: [`docs/PLAN.md`](docs/PLAN.md)

---

## Quick Start

```powershell
# Launch Traum Engine (opens the editable Floating Diorama Island!)
cargo run --bin traum-engine

# Run all CPU + headless GPU shader/compute/render verification tests
cargo test
cargo run --bin traum-engine -- --self-test
```

---

## Controls Across All Devices

### 1. Mouse & Keyboard
| Action | Control |
| :--- | :--- |
| **Sculpt / Paint / Grab Toy** | `Left Click + Drag` (`Alt + Left Drag` quick-carves!) |
| **Orbit / Pan / Zoom Camera** | `Right-Drag` (Orbit), `Middle-Drag` or `Shift + Right-Drag` (Pan), `Scroll` (Zoom), `WASD` / `Q` / `E` |
| **Bouncy Radial Pie Wheel** | Hold `Space` -> flick mouse -> release! |
| **Brush Size & Smooth Blend** | `Shift + Scroll` or `[` / `]` (Size), `Ctrl + Scroll` (Smooth Blend) |
| **Modes** | `1` Sculpt · `2` Paint · `3` Animate · `4` or `Tab` **Play!** |
| **Shapes / Flecks / Symmetry** | `F` Next Shape · `G` Next Living Fleck Style · `M` Mirror X Symmetry |
| **Animate Toys Live** | `P` Spawn Toy Prop · `R` Arm Live Action Recorder -> drag prop to record loop! |
| **Undo / Redo / Save / Load** | `Ctrl+Z` (or `Z`) Undo · `Ctrl+Y` Redo · `Ctrl+S` / `F5` Save · `F9` Load |
| **Play Mode** | `WASD` / Arrows Walk · `Space` Jump · `Right-Drag` Camera · `Tab` / `Esc` Back to Sculpt |

### 2. Multi-Touch / Tablet
| Action | Gesture |
| :--- | :--- |
| **Sculpt / Paint / Tap UI Pills** | `1-Finger Drag / Tap` |
| **Orbit & Pinch-Zoom Camera** | `2-Finger Drag` (Orbit) + `2-Finger Pinch` (Zoom) — auto-cancels accidental 1-finger start! |
| **Quick Undo / Redo** | `2-Finger Tap` (Undo) · `3-Finger Tap` (Redo) |
| **Radial Pie Menu** | Tap top `Wheel` button -> tap any slice |
| **Play Mode** | Left screen side = Virtual Analog Stick · Bottom-right tap = Jump · Right drag = Orbit |

### 3. Game Controller (Xbox / DualSense / Switch Pro)
| Action | Control |
| :--- | :--- |
| **3D Imp & Camera** | `Right Stick` moves 3D Imp · `Left Stick` orbits diorama |
| **Analog Sculpt / Carve** | `R2` Analog Trigger (pull depth controls brush size!) · `L2 + R2` Carve |
| **Radial Pie Wheel** | Hold `L1 / LB` + flick `Right Stick` -> release `L1` |
| **Quick Cycle & Undo** | `Y / Triangle` Shape · `B / Circle` Fleck · `X / Square` Mirror · `D-Pad Left/Right` Undo/Redo · `D-Pad Up/Down` Brush Size |
| **Toggle Play Mode** | `Start` (`Left Stick` Walk · `Right Stick` Camera · `A / Cross` Jump!) |
