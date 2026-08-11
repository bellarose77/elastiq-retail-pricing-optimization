export default function Icon({ name, size = 18 }) {
  const paths = {
    overview: <><path d="M4 4h7v7H4z"/><path d="M13 4h7v4h-7z"/><path d="M13 10h7v10h-7z"/><path d="M4 13h7v7H4z"/></>,
    decisions: <><path d="M4 6h16"/><path d="M4 12h16"/><path d="M4 18h10"/><circle cx="18" cy="18" r="2"/></>,
    sensitivity: <><path d="M4 18V6"/><path d="M10 18V10"/><path d="M16 18V4"/><path d="M22 18H2"/></>,
    validation: <><path d="M12 22s8-3 8-10V5l-8-3-8 3v7c0 7 8 10 8 10z"/><path d="m8 12 3 3 5-6"/></>,
    methods: <><circle cx="6" cy="6" r="3"/><circle cx="18" cy="6" r="3"/><circle cx="12" cy="18" r="3"/><path d="m8.5 8 2.3 6.2M15.5 8l-2.3 6.2M9 6h6"/></>,
    data: <><ellipse cx="12" cy="5" rx="8" ry="3"/><path d="M4 5v6c0 1.7 3.6 3 8 3s8-1.3 8-3V5"/><path d="M4 11v6c0 1.7 3.6 3 8 3s8-1.3 8-3v-6"/></>,
    report: <><path d="M6 2h9l5 5v15H6z"/><path d="M14 2v6h6M9 13h8M9 17h8M9 9h2"/></>,
    guide: <><path d="M4 5a4 4 0 0 1 4-2h4v18H8a4 4 0 0 0-4 2z"/><path d="M20 5a4 4 0 0 0-4-2h-4v18h4a4 4 0 0 1 4 2z"/></>,
    settings: <><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.7 1.7 0 0 0 .3 1.9l.1.1-2.8 2.8-.1-.1a1.7 1.7 0 0 0-1.9-.3 1.7 1.7 0 0 0-1 1.6v.2h-4V21a1.7 1.7 0 0 0-1-1.6 1.7 1.7 0 0 0-1.9.3l-.1.1L4.2 17l.1-.1a1.7 1.7 0 0 0 .3-1.9A1.7 1.7 0 0 0 3 14H2.8v-4H3a1.7 1.7 0 0 0 1.6-1 1.7 1.7 0 0 0-.3-1.9L4.2 7 7 4.2l.1.1a1.7 1.7 0 0 0 1.9.3A1.7 1.7 0 0 0 10 3V2.8h4V3a1.7 1.7 0 0 0 1 1.6 1.7 1.7 0 0 0 1.9-.3l.1-.1L19.8 7l-.1.1a1.7 1.7 0 0 0-.3 1.9 1.7 1.7 0 0 0 1.6 1h.2v4H21a1.7 1.7 0 0 0-1.6 1z"/></>,
    download: <><path d="M12 3v12"/><path d="m7 10 5 5 5-5"/><path d="M4 21h16"/></>,
    arrow: <><path d="M5 12h14"/><path d="m14 7 5 5-5 5"/></>,
    close: <><path d="m6 6 12 12M18 6 6 18"/></>,
    spark: <><path d="m12 2 1.8 5.2L19 9l-5.2 1.8L12 16l-1.8-5.2L5 9l5.2-1.8z"/><path d="m19 15 .8 2.2L22 18l-2.2.8L19 21l-.8-2.2L16 18l2.2-.8z"/></>,
    live: <><path d="M4 18V6"/><path d="M4 18h16"/><path d="m6 15 4-5 3 3 5-7"/><circle cx="18" cy="6" r="1.5"/></>,
    play: <><path d="M8 5v14l11-7z"/></>,
  };
  return <svg aria-hidden="true" width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round">{paths[name] || paths.overview}</svg>;
}
