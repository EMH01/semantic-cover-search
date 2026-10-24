"""Presentation-only styling for the Gradio app.

Keep visual tokens and layout CSS separate from data and inference code.
"""

APP_CSS = """
:root {
    --scs-ink: #19263d;
    --scs-muted: #5b6880;
    --scs-border: #e1e6f0;
    --scs-violet: #6254da;
    --scs-soft: #f7f8fc;
}

.gradio-container {
    max-width: 1240px !important;
    margin: 0 auto !important;
    padding: 24px 24px 56px !important;
    font-family: Inter, ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif;
}

.gradio-container .scs-hero {
    background: linear-gradient(115deg, #111d36 0%, #282b61 66%, #45428b 100%);
    border: 1px solid #3f4a76;
    color: #f9faff;
    border-radius: 22px;
    padding: 28px 34px 30px;
    margin-bottom: 22px;
    box-shadow: 0 12px 35px rgba(28, 36, 83, .13);
}
.scs-hero .scs-eyebrow {
    display: inline-flex;
    gap: 8px;
    align-items: center;
    letter-spacing: .13em;
    text-transform: uppercase;
    font-weight: 650;
    font-size: .72rem;
    color: #c9ceff;
    margin-bottom: 10px;
}
.scs-hero .scs-brand {
    font-size: clamp(1.55rem, 3vw, 2.18rem);
    line-height: 1.19;
    letter-spacing: -.035em;
    font-weight: 760;
    color: #fff;
    margin: 0 0 9px;
}
.scs-hero .scs-tagline {
    font-size: .96rem;
    line-height: 1.65;
    color: #e0e4ff;
    max-width: 700px;
    margin: 0;
}
.scs-hero .scs-capabilities {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin-top: 20px;
}
.scs-hero .scs-capabilities span {
    border: 1px solid rgba(226, 230, 255, .28);
    background: rgba(255, 255, 255, .085);
    padding: 6px 11px;
    border-radius: 99px;
    font-size: .76rem;
    letter-spacing: .015em;
    color: #f5f5ff;
}

.gradio-container [role="tablist"] {
    gap: 6px;
    border-bottom: 1px solid #e3e7f1;
    margin-bottom: 18px;
}
.gradio-container button[role="tab"] {
    color: #5f6b82;
    font-weight: 650;
    border-radius: 9px 9px 0 0;
    padding: 12px 17px;
}
.gradio-container button[role="tab"][aria-selected="true"] {
    color: #5044c0;
    border-bottom-color: #6254da;
}
.gradio-container .scs-panel,
.gradio-container .scs-match,
.gradio-container .scs-video {
    color: var(--scs-ink);
}
.gradio-container input:focus-visible,
.gradio-container button:focus-visible,
.gradio-container a:focus-visible {
    outline: 2px solid #8178e4;
    outline-offset: 2px;
}

.gradio-container button.scs-primary {
    background: #6154d9 !important;
    border-color: #6154d9 !important;
    color: #fff !important;
    border-radius: 11px !important;
    font-weight: 650 !important;
    min-height: 44px;
    box-shadow: 0 4px 12px rgba(94, 78, 204, .14);
}
.gradio-container button.scs-primary:hover {
    background: #5145c2 !important;
    border-color: #5145c2 !important;
}
.gradio-container button.scs-secondary {
    border: 1px solid #d4d9ec !important;
    background: #fff !important;
    color: #493eaa !important;
    border-radius: 11px !important;
    font-weight: 650 !important;
    min-height: 42px;
}
.gradio-container button.scs-secondary:hover {
    background: #f4f2ff !important;
}

.gradio-container .scs-section-title h2,
.gradio-container .scs-section-title h3 {
    color: var(--scs-ink);
    letter-spacing: -.02em;
    font-weight: 730;
    margin: 14px 0 4px;
}
.gradio-container .scs-intro {
    color: var(--scs-muted);
    font-size: .93rem;
    line-height: 1.6;
}
.gradio-container .scs-panel {
    background: #fff;
    border: 1px solid var(--scs-border);
    padding: 20px !important;
    border-radius: 16px;
    box-shadow: 0 3px 13px rgba(31, 42, 73, .035);
}
.gradio-container .scs-panel input,
.gradio-container .scs-panel textarea {
    border-radius: 9px;
}
/* Reserve a dedicated, non-overlapping area for the one native loading indicator. */
.gradio-container .scs-status {
    position: relative;
    isolation: isolate;
    min-height: 52px;
    padding: 12px 2px 16px;
    margin: 8px 0 18px;
    overflow-wrap: anywhere;
}
.gradio-container .scs-status p {
    margin: 0;
    line-height: 1.6;
}
.gradio-container .scs-dataframe {
    border-radius: 13px;
    overflow: hidden;
}
.gradio-container .scs-dataframe th {
    font-family: inherit;
    font-weight: 700;
    color: #34425c;
}

.scs-video-grid {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 14px;
    margin: 12px 0 20px;
}
.scs-video {
    display: grid;
    grid-template-columns: 158px minmax(0, 1fr);
    gap: 15px;
    min-width: 0;
    border: 1px solid var(--scs-border);
    border-radius: 15px;
    padding: 12px;
    background: #fff;
    box-shadow: 0 3px 12px rgba(21, 33, 69, .035);
    transition: border-color .15s ease, transform .15s ease;
}
.scs-video:hover { border-color: #a8a4f4; transform: translateY(-1px); }
.scs-video .scs-video-thumb {
    aspect-ratio: 16 / 9;
    width: 100%;
    border-radius: 10px;
    background: linear-gradient(135deg, #232e52, #6764b1);
    object-fit: cover;
    display: block;
}
.scs-video-body { min-width: 0; display: flex; flex-direction: column; }
.scs-video-title {
    color: #1b2740;
    font-weight: 700;
    font-size: .9rem;
    line-height: 1.42;
    margin: 2px 0 7px;
    overflow-wrap: anywhere;
}
.scs-video-channel { color: var(--scs-muted); font-size: .78rem; margin-bottom: 13px; }
.scs-video-link {
    display: inline-flex;
    align-items: center;
    width: fit-content;
    margin-top: auto;
    font-size: .82rem;
    text-decoration: none !important;
    color: #4e43c0 !important;
    font-weight: 700;
}
.scs-video-link:hover { text-decoration: underline !important; }
.scs-empty {
    border: 1px dashed #bec6da;
    background: #f9faff;
    border-radius: 14px;
    padding: 23px;
    color: var(--scs-muted);
}

.scs-match-grid {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 14px;
    margin-top: 16px;
}
.scs-match {
    border: 1px solid var(--scs-border);
    border-radius: 15px;
    padding: 18px;
    background: #fff;
    min-width: 0;
}
.scs-match-head {
    color: #6b728b;
    font-size: .72rem;
    font-weight: 700;
    letter-spacing: .08em;
    text-transform: uppercase;
    margin: 0 0 12px;
}
.scs-track-name {
    font-size: .92rem;
    font-weight: 710;
    line-height: 1.42;
    color: #1b2740;
    overflow-wrap: anywhere;
}
.scs-track-source { font-size: .75rem; color: #717d92; margin-top: 3px; }
.scs-match-divider { border-top: 1px dashed #d7dded; margin: 13px 0; }
.scs-metrics { display: flex; flex-wrap: wrap; gap: 12px; margin-top: 18px; }
.scs-metric { min-width: 115px; flex: 1; }
.scs-metric-label { color: #717d92; font-size: .7rem; margin-bottom: 5px; }
.scs-metric-value { font-size: 1.25rem; font-weight: 770; color: #252e4b; }
.scs-meter {
    background: #edf0f9;
    border-radius: 30px;
    height: 5px;
    overflow: hidden;
    margin-top: 7px;
}
.scs-meter-fill { height: 100%; border-radius: 30px; background: #6254da; }
.scs-meter-fill-alt { background: #7a90c4; }
.scs-match-note { margin: 12px 0 0; font-size: .77rem; color: #747b92; line-height: 1.5; }

@media (max-width: 900px) {
    .scs-video-grid, .scs-match-grid { grid-template-columns: 1fr; }
}
@media (max-width: 600px) {
    .gradio-container { padding: 10px 12px 38px !important; }
    .gradio-container .scs-hero { padding: 23px 20px; border-radius: 16px; }
    .gradio-container .scs-panel { padding: 14px !important; }
    .scs-video { grid-template-columns: 115px minmax(0, 1fr); gap: 11px; }
    .scs-video-title { font-size: .82rem; }
}
@media (prefers-reduced-motion: reduce) {
    .scs-video { transition: none; }
    .scs-video:hover { transform: none; }
}
"""

HERO_HTML = """
<header class="scs-hero">
    <div class="scs-eyebrow">♪ &nbsp; Music discovery · Lyrics analysis</div>
    <h1 class="scs-brand">Semantic Cover Search</h1>
    <p class="scs-tagline">Explore recordings on YouTube, build a temporary song
    catalog, and compare the words shared across different performances.</p>
    <div class="scs-capabilities" aria-label="Application capabilities">
        <span>YouTube discovery</span>
        <span>Lyric similarity</span>
        <span>Local AI</span>
    </div>
</header>
"""
