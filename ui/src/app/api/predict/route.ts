// Server-side proxy route that forwards prediction requests to the model API.
// This keeps the model API URL out of the browser and allows the UI server
// (deployed in the same GCP project) to reach an internal Cloud Run service.

import { NextResponse } from 'next/server'

export async function POST(req: Request) {
  try {
    const body = await req.json();

    // Prefer a server-only env var if available; fall back to the public one.
    const modelUrl = process.env.MODEL_API || process.env.NEXT_PUBLIC_MODEL_API;
    if (!modelUrl) {
      return NextResponse.json({ error: 'Model API URL not configured' }, { status: 500 });
    }

    const resp = await fetch(modelUrl, {
      method: 'POST',
      headers: {
        'Accept': 'application/json',
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(body),
    });

    const contentType = resp.headers.get('content-type') || '';
    const text = await resp.text();

    if (contentType.includes('application/json')) {
      return new Response(text, {
        status: resp.status,
        headers: { 'content-type': 'application/json' },
      });
    }

    console.error('Upstream model API returned non-JSON response', {
      modelUrl,
      status: resp.status,
      statusText: resp.statusText,
      contentType,
      body: text,
    });

    return NextResponse.json(
      { error: `Model API returned non-JSON response: ${resp.status} ${resp.statusText}` },
      { status: 502 },
    );
  } catch (err) {
    console.error('Proxy predict error:', err);
    return NextResponse.json({ error: 'Proxy error' }, { status: 500 });
  }
}
