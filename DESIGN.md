---
version: alpha
name: VisionOps
 description: A light control-room aesthetic for an image-aware SRE agent. Designed to feel like a clean Network Operations Center dashboard — high contrast, data-dense, and fast.
colors:
  background: "#F8F8F8"
  surface: "#FFFFFF"
  surface-raised: "#F2F2F2"
  border: "#D0D0D0"
  primary: "#4ECDC4"
  primary-dim: "#3BA99E"
  secondary: "#FF4D6D"
  tertiary: "#FFD700"
  success: "#00FF41"
  warning: "#FFB800"
  text: "#1A1A1A"
  text-muted: "#666666"
  on-primary: "#1A1A1A"
  on-secondary: "#FFFFFF"
  on-tertiary: "#1A1A1A"
  on-success: "#1A1A1A"
  on-warning: "#1A1A1A"
typography:
  h1:
    fontFamily: "Inter"
    fontSize: "2.25rem"
    fontWeight: 700
    letterSpacing: "-0.02em"
    lineHeight: 1.1
  h2:
    fontFamily: "Inter"
    fontSize: "1.5rem"
    fontWeight: 600
    letterSpacing: "-0.01em"
    lineHeight: 1.2
  h3:
    fontFamily: "Inter"
    fontSize: "1rem"
    fontWeight: 600
    letterSpacing: "0"
    lineHeight: 1.3
  body:
    fontFamily: "Inter"
    fontSize: "0.875rem"
    fontWeight: 400
    lineHeight: 1.5
  body-sm:
    fontFamily: "Inter"
    fontSize: "0.75rem"
    fontWeight: 400
    lineHeight: 1.5
  mono:
    fontFamily: "JetBrains Mono"
    fontSize: "0.8125rem"
    fontWeight: 400
    lineHeight: 1.5
  label:
    fontFamily: "Inter"
    fontSize: "0.6875rem"
    fontWeight: 600
    letterSpacing: "0.08em"
    lineHeight: 1.2
    textTransform: uppercase
rounded:
  sm: 4px
  md: 8px
  lg: 12px
  full: 9999px
spacing:
  xs: 4px
  sm: 8px
  md: 16px
  lg: 24px
  xl: 32px
  xxl: 48px
shadows:
  panel: "0 4px 24px rgba(0, 0, 0, 0.45)"
  glow-primary: "0 0 16px rgba(0, 240, 255, 0.25)"
  glow-alert: "0 0 16px rgba(255, 77, 109, 0.25)"
components:
  panel:
    backgroundColor: "{colors.surface}"
    borderColor: "{colors.border}"
    borderWidth: "1px"
    borderRadius: "{rounded.md}"
    padding: "{spacing.md}"
    shadow: "{shadows.panel}"
  panel-raised:
    backgroundColor: "{colors.surface-raised}"
    borderColor: "{colors.border}"
    borderWidth: "1px"
    borderRadius: "{rounded.md}"
    padding: "{spacing.md}"
    shadow: "{shadows.panel}"
  button-primary:
    backgroundColor: "{colors.primary}"
    textColor: "{colors.on-primary}"
    borderRadius: "{rounded.sm}"
    padding: "10px 16px"
    font: "{typography.label}"
    shadow: "{shadows.glow-primary}"
  button-primary-hover:
    backgroundColor: "{colors.primary-dim}"
  button-secondary:
    backgroundColor: "transparent"
    textColor: "{colors.text}"
    borderColor: "{colors.border}"
    borderWidth: "1px"
    borderRadius: "{rounded.sm}"
    padding: "10px 16px"
    font: "{typography.label}"
  button-danger:
    backgroundColor: "{colors.secondary}"
    textColor: "{colors.on-secondary}"
    borderRadius: "{rounded.sm}"
    padding: "10px 16px"
    font: "{typography.label}"
    shadow: "{shadows.glow-alert}"
  input:
    backgroundColor: "{colors.surface-raised}"
    borderColor: "{colors.border}"
    borderWidth: "1px"
    borderRadius: "{rounded.sm}"
    padding: "10px 12px"
    textColor: "{colors.text}"
    font: "{typography.body}"
  badge:
    borderRadius: "{rounded.full}"
    padding: "4px 10px"
    font: "{typography.label}"
  badge-success:
    backgroundColor: "{colors.success}"
    textColor: "{colors.on-success}"
  badge-warning:
    backgroundColor: "{colors.warning}"
    textColor: "{colors.on-warning}"
  badge-alert:
    backgroundColor: "{colors.secondary}"
    textColor: "{colors.on-secondary}"
  log-info:
    textColor: "{colors.text-muted}"
    font: "{typography.mono}"
  log-action:
    textColor: "{colors.primary}"
    font: "{typography.mono}"
  log-alert:
    textColor: "{colors.secondary}"
    font: "{typography.mono}"
  log-success:
    textColor: "{colors.success}"
    font: "{typography.mono}"
---

## Overview

VisionOps is an image-aware SRE agent dashboard. The visual identity is inspired by clean Network Operations Centers (NOCs): light, high-contrast, and information-dense. The design prioritizes readability under pressure, clear status signaling, and a sense of speed.

The UI should feel like a mission-critical tool — not a consumer app. Every element exists to surface signal and reduce time-to-resolution.

## Colors

The palette is built around a clean white background so that screenshots, charts, and agent outputs stand out. Teal is the primary action color; it reads as "system active" without the urgency of red. Red is reserved for real alerts. Gold highlights warnings and key metrics. Green confirms success.

- **background (#F8F8F8):** Clean foundation. Keeps screenshots and telemetry as the heroes.
- **surface (#FFFFFF):** Primary panel background.
- **surface-raised (#F2F2F2):** Inputs, hovered rows, and secondary surfaces.
- **border (#D0D0D0):** Subtle separation without visual noise.
- **primary (#4ECDC4):** Teal for active systems, primary actions, and agent thought traces.
- **secondary (#FF4D6D):** Alert red for incidents, failures, and destructive actions.
- **tertiary (#FFD700):** Gold for warnings, attention markers, and highlighted metrics.
- **success (#00FF41):** Confirmation green for resolved states and healthy signals.
- **text (#1A1A1A):** High-contrast near-black for primary text.
- **text-muted (#666666):** Captions, timestamps, and metadata.

## Typography

Inter keeps the interface human-readable at small sizes. JetBrains Mono is used for logs, code, metrics, and any agent-structured output.

- **h1:** Large, tight headline for the hero/dashboard title.
- **h2:** Section headers inside panels.
- **h3:** Card titles and subsection labels.
- **body:** Default UI copy.
- **body-sm:** Captions and helper text.
- **mono:** Logs, token counts, latency values, JSON output.
- **label:** Uppercase, spaced labels for buttons, badges, and section headers.

## Layout

Use a two-column dashboard layout on desktop:

- **Left column (40%):** Screenshot input / live feed, upload controls, and incident metadata.
- **Right column (60%):** Agent reasoning stream, structured diagnosis, suggested actions, and live telemetry gauges.

On mobile, stack columns vertically with the screenshot at the top.

Grid spacing follows an 8px base. Panels use consistent internal padding and a 1px border. Avoid rounded corners larger than 12px — the aesthetic is sharp and technical.

## Elevation & Depth

Depth is expressed through shadow and layered surfaces, not heavy borders.

- Panels sit on **surface** over **background**.
- Raised inputs and cards use **surface-raised**.
- Active elements (running agent, primary buttons) receive a soft teal glow.
- Alert elements receive a soft red glow.

## Shapes

- Panels, cards, and inputs: **8px radius** (`rounded.md`).
- Buttons and tags: **4px radius** (`rounded.sm`).
- Status badges and avatars: **fully rounded** (`rounded.full`).

## Components

### Panel
The basic building block. Use for the screenshot viewer, agent log, diagnosis card, and telemetry gauges.

### Button Primary
Use for the main action (e.g., "Analyze Screenshot", "Run Remediation"). Cyan with a subtle glow.

### Button Secondary
Use for auxiliary actions (e.g., "Upload New", "Clear Log").

### Button Danger
Use for destructive or high-impact actions (e.g., "Restart Service", "Kill Pod").

### Input
File upload drop zone and text inputs. Keep borders visible but not heavy.

### Badge
Status pills: `healthy`, `warning`, `critical`, `analyzing`, `resolved`. Fully rounded, uppercase label text.

### Log Entries
Monospace by default. Color-code by event type:
- `log-info` — neutral system events and timestamps
- `log-action` — agent actions and tool calls
- `log-alert` — incidents and failures
- `log-success` — resolutions and confirmations

## Do's and Don'ts

- **Do** keep the background light so screenshots and telemetry remain the focal points.
- **Do** use teal sparingly as the active/agent color.
- **Do** reserve red for real problems and destructive actions.
- **Do** use monospace for any structured or machine-generated output.
- **Don't** use pure black or pure white — the palette is intentionally muted for reduced eye strain.
- **Don't** add decorative gradients or illustrations. The design is flat, functional, and fast.
- **Don't** use more than one primary action per panel.
