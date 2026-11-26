# Traum Engine

A playful 3D Signed Distance Field (SDF) & Painterly Brushstroke creation engine built in **Rust** and **`wgpu`**.

Sculpt worlds out of warm, tactile SDF clay coated in living painterly brush dabs, **Edit Inside** custom Toy Objects, create **Linked Copies** across your island, link multi-part puppets with spring & hinge joints, animate them live by waving them around, and jump straight into **Play Mode**—across **Mouse & Keyboard**, **Multi-Touch**, and **Game Controllers**.

Full user manual: [`docs/manual.html`](docs/manual.html) · Architecture & roadmap: [`docs/PLAN.md`](docs/PLAN.md)

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
| **Edit Inside Toy / Exit Toy** | Hover Toy + `Enter` (or `V` / `Edit Inside Toy` button) · `Enter` / `Esc` Exit |
| **Linked Copy & Unlink Copy** | `C` or `Ctrl+D` Linked Copy (edits update all copies!) · `U` Unlink Copy |
| **Multi-Part Puppet Limbs** | `L` or `Attach Limb` attaches a Spring/Hinge-jointed child part! |
| **3D Rotate & Scale Handles** | In `Animate` mode (`3`), drag the **Turquoise Ring** (Rotate) or **Gold Orb** (Scale) |
| **Orbit / Pan / Zoom Camera** | `Right-Drag` (Orbit), `Middle-Drag` or `Shift + Right-Drag` (Pan), `Scroll` (Zoom), `WASD` / `Q` / `E` |
| **Bouncy Radial Pie Wheel** | Hold `Space` -> flick mouse -> release! |
| **Brush Size & Soft Blend** | `Shift + Scroll` or `[` / `]` (Size / Toy Scale), `Ctrl + Scroll` (Soft Blend) |
| **Modes** | `1` Sculpt · `2` Paint · `3` Animate · `4` or `Tab` **Play!** |
| **Shapes / Brush Styles / Symmetry** | `F` Next Shape · `G` Next Living Brush Style · `M` Mirror X (Local inside Toys!) |
| **Animate Toys Live** | `P` New Toy Object · `R` Arm Live Action Recorder -> drag Toy to record loop! |
| **Undo / Redo / Save / Load** | `Ctrl+Z` (or `Z`) Undo · `Ctrl+Y` Redo · `Delete` Remove Toy · `Ctrl+S` / `F5` Save · `F9` Load |
| **Play Mode** | `WASD` / Arrows Walk (rides moving platforms!) · `Space` Jump · `Right-Drag` Camera · `Tab` / `Esc` Sculpt |

### 2. Multi-Touch / Tablet
| Action | Gesture |
| :--- | :--- |
| **Sculpt / Paint / Drag 3D Handles** | `1-Finger Drag / Tap` |
| **Edit Inside / Copy / Attach Limb** | Tap `Edit Inside Toy`, `Copy Linked Toy`, or `Attach Limb` buttons (or Radial Wheel!) |
| **Orbit & Pinch-Zoom Camera** | `2-Finger Drag` (Orbit) + `2-Finger Pinch` (Zoom) — auto-cancels accidental 1-finger start! |
| **Quick Undo / Redo** | `2-Finger Tap` (Undo) · `3-Finger Tap` (Redo) |
| **Radial Pie Menu** | Tap top `Menu` button -> tap any slice |
| **Play Mode** | Left screen side = Virtual Analog Stick · Bottom-right tap = Jump · Right drag = Orbit |

### 3. Game Controller (Xbox / DualSense / Switch Pro)
| Action | Control |
| :--- | :--- |
| **3D Spark & Camera** | `Right Stick` moves 3D Spark cursor · `Left Stick` orbits diorama |
| **Analog Sculpt / Carve** | `R2` Analog Trigger (pull depth controls brush size!) · `L2 + R2` Carve |
| **Edit Inside & Linked Copy** | `R1 / RB` Enter / Exit Toy · `Select` Linked Copy |
| **Radial Pie Wheel** | Hold `L1 / LB` + flick `Right Stick` -> release `L1` |
| **Quick Cycle & Undo** | `Y / Triangle` Shape · `B / Circle` Brush Style · `X / Square` Mirror · `D-Pad Left/Right` Undo/Redo · `D-Pad Up/Down` Size |
| **Toggle Play Mode** | `Start` (`Left Stick` Walk · `Right Stick` Camera · `A / Cross` Jump!) |
