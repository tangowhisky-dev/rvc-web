# RVC Web Frontend

Next.js 15 (App Router) + React 19 + TypeScript + Tailwind CSS 4.

## Structure

```
app/
  layout.tsx              ← Root layout, font, global styles
  page.tsx                ← Main online dashboard (profiles, training, offline)
  offline/
    page.tsx              ← Offline inference page (file upload + convert)
  realtime/
    page.tsx              ← Realtime voice conversion page (mic input)
  globals.css             ← Tailwind base, theme variables, custom utilities
components/
  ui/                     ← Reusable UI primitives (buttons, inputs, modals, etc.)
  ProfileCard.tsx         ← Profile list item with stats
  TrainingProgress.tsx    ← Live training progress with WebSocket updates
  OfflineConvert.tsx      ← Offline conversion form + controls
  RealtimePanel.tsx       ← Realtime inference controls
lib/
  api.ts                  ← API client (fetch wrappers, type-safe endpoints)
  audio.ts                ← Audio utilities (WAV encoding, PCM processing)
  realtime.ts             ← Realtime WebSocket client
```

## Key Features

- **Profile management**: Create, upload audio, view F0 stats (mean, std, percentiles, velocity)
- **Training**: Start/cancel training jobs, live WebSocket progress, epoch loss charts
- **Offline inference**: Upload audio file, configure pitch/autotune/protect, download result
- **Realtime inference**: Mic input → voice conversion → playback via Web Audio API
- **Engine selection**: RVC or Beatrice 2 at profile creation (immutable)

## API Integration

All API calls go through `lib/api.ts` which wraps `fetch()` with:
- Base URL resolution (`/api/` relative paths)
- Error handling with typed responses
- File upload support (FormData)

WebSocket connections for training progress and realtime inference are managed by dedicated clients in `lib/realtime.ts`.

## Styling

Tailwind CSS 4 with CSS variables for theming. Dark mode via class strategy. Custom utility classes in `globals.css` for animations and layout patterns.

## Development

```bash
cd frontend
npm install
npm run dev        # localhost:3000
npm run build      # production build
npm run lint       # ESLint + TypeScript
```

The dev server proxies `/api/` requests to the backend (configured in `next.config.ts`).
