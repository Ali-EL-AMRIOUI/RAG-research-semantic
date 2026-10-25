# Frontend — Semantic RAG UI

Next.js chat interface: multi-file upload, persistent multi-turn conversation, cited sources per answer. See the [root README](../README.md) for the full picture.

## Setup

```bash
npm install
echo "NEXT_PUBLIC_API_URL=http://127.0.0.1:8000" > .env.local
npm run dev
```

> Use `127.0.0.1`, not `localhost`, in `NEXT_PUBLIC_API_URL`. Some browsers resolve `localhost` to IPv6 first and stall on it, causing requests to time out even though the backend responds instantly over `curl`.

Requires the backend running separately (see [`backend/README.md`](../backend/README.md)).

## Structure

app/
├── page.tsx # main chat UI — upload, conversation, sources
└── layout.tsx
lib/
└── api.ts # centralized axios client (base URL, API key header, timeout)


## Scripts

```bash
npm run dev      # dev server, hot reload
npm run build    # production build — NEXT_PUBLIC_* vars are inlined here, not at runtime
npm run start    # serve the production build
npm run lint     # eslint
```

## Docker build-time variables

`NEXT_PUBLIC_API_URL` and `NEXT_PUBLIC_API_KEY` are baked into the JS bundle at `npm run build` time. When building via Docker, they must be passed as build args (already wired in `docker-compose.yml`); changing them requires a rebuild (`docker compose build --no-cache frontend`), not just a restart.

## A note on the API key

`NEXT_PUBLIC_API_KEY`, if set, is visible in the browser bundle and network requests — it stops anonymous bots, not a motivated user of the app itself. See [root README](../README.md) for the full caveat and mitigation options.

