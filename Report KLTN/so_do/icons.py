"""Bộ biểu tượng SVG phẳng, nhúng thẳng vào draw.io (data URI) nên không phụ thuộc thư viện ngoài."""
import base64


def _net():
    cols = [(10, [16, 32, 48]), (32, [12, 25, 39, 52]), (54, [22, 42])]
    lines = []
    for (x1, ys1), (x2, ys2) in zip(cols, cols[1:]):
        lines += [f"<line x1='{x1}' y1='{a}' x2='{x2}' y2='{b}'/>" for a in ys1 for b in ys2]
    dots = [f"<circle cx='{x}' cy='{y}' r='5.5'/>" for x, ys in cols for y in ys]
    return (f"<g stroke='#4DB6AC' stroke-width='1.6'>{''.join(lines)}</g>"
            f"<g fill='#00897B' stroke='#fff' stroke-width='1.5'>{''.join(dots)}</g>")


def _share():
    pts = [(10, 12), (54, 12), (10, 52), (54, 52)]
    ln = "".join(f"<line x1='32' y1='32' x2='{x}' y2='{y}'/>" for x, y in pts)
    dt = "".join(f"<circle cx='{x}' cy='{y}' r='7' fill='#FFB300'/>" for x, y in pts)
    return f"<g stroke='#8D6E63' stroke-width='3'>{ln}</g>{dt}<circle cx='32' cy='32' r='11' fill='#E65100'/>"


SVG = {
    "db": "<ellipse cx='32' cy='13' rx='20' ry='7' fill='#64B5F6'/><path d='M12 13v38c0 4 9 7 20 7s20-3 20-7V13c0 4-9 7-20 7s-20-3-20-7z' fill='#1E6FC2'/><path d='M12 26c0 4 9 7 20 7s20-3 20-7M12 39c0 4 9 7 20 7s20-3 20-7' stroke='#90CAF9' stroke-width='2' fill='none'/>",
    "warehouse": "<path d='M3 27 32 8l29 19v4H3z' fill='#C0392B'/><rect x='8' y='31' width='48' height='27' fill='#ECEFF4' stroke='#7A8CA5' stroke-width='2'/><rect x='22' y='38' width='20' height='20' fill='#7A8CA5'/><path d='M22 43h20M22 48h20M22 53h20' stroke='#ECEFF4' stroke-width='1.5'/><rect x='10' y='45' width='10' height='13' fill='#F0A040'/><rect x='44' y='45' width='10' height='13' fill='#F0A040'/>",
    "truck": "<rect x='3' y='16' width='38' height='28' rx='2' fill='#F0A040'/><path d='M41 24h12l8 11v9H41z' fill='#5B6B82'/><rect x='45' y='27' width='8' height='7' fill='#D6E4F5'/><circle cx='14' cy='47' r='6.5' fill='#2D3748'/><circle cx='14' cy='47' r='2.5' fill='#fff'/><circle cx='50' cy='47' r='6.5' fill='#2D3748'/><circle cx='50' cy='47' r='2.5' fill='#fff'/>",
    "people": "<circle cx='17' cy='19' r='8' fill='#B39DDB'/><circle cx='47' cy='19' r='8' fill='#B39DDB'/><path d='M3 50c0-10 6-16 14-16s14 6 14 16z' fill='#B39DDB'/><path d='M33 50c0-10 6-16 14-16s14 6 14 16z' fill='#B39DDB'/><circle cx='32' cy='24' r='10' fill='#673AB7'/><path d='M13 60c0-12 8-21 19-21s19 9 19 21z' fill='#673AB7'/>",
    "user": "<circle cx='32' cy='19' r='12' fill='#1E88E5'/><path d='M9 60c0-15 10-25 23-25s23 10 23 25z' fill='#1E88E5'/>",
    "calendar": "<rect x='8' y='12' width='48' height='44' rx='5' fill='#fff' stroke='#E53935' stroke-width='3'/><rect x='8' y='12' width='48' height='12' rx='4' fill='#E53935'/><rect x='18' y='6' width='4' height='12' rx='2' fill='#555'/><rect x='42' y='6' width='4' height='12' rx='2' fill='#555'/><g fill='#EF9A9A'><rect x='16' y='30' width='7' height='6'/><rect x='28' y='30' width='7' height='6'/><rect x='40' y='30' width='7' height='6'/><rect x='16' y='42' width='7' height='6'/><rect x='28' y='42' width='7' height='6'/></g>",
    "tag": "<path d='M6 34 30 10h22v22L28 56z' fill='#43A047'/><circle cx='43' cy='19' r='4' fill='#fff'/><text x='28' y='41' font-size='17' font-family='Arial' font-weight='bold' fill='#fff' text-anchor='middle'>$</text>",
    "chart": "<rect x='6' y='8' width='52' height='48' rx='4' fill='#fff' stroke='#90A4AE' stroke-width='2'/><polyline points='11,46 20,34 28,40 38,22 46,30 53,16' fill='none' stroke='#1E88E5' stroke-width='3.5' stroke-linejoin='round'/>",
    "forecast": "<path d='M8 8v48h50' stroke='#78909C' stroke-width='3' fill='none'/><polyline points='12,46 20,38 28,42 36,28' fill='none' stroke='#1E88E5' stroke-width='3.5' stroke-linejoin='round'/><polyline points='36,28 46,22 56,12' fill='none' stroke='#FB8C00' stroke-width='3.5' stroke-dasharray='5 3'/><circle cx='36' cy='28' r='3.5' fill='#1E88E5'/>",
    "robot": "<rect x='30' y='4' width='4' height='9' fill='#5E35B1'/><circle cx='32' cy='5' r='3.5' fill='#FFB300'/><rect x='12' y='12' width='40' height='30' rx='8' fill='#7E57C2'/><circle cx='24' cy='27' r='5' fill='#fff'/><circle cx='40' cy='27' r='5' fill='#fff'/><circle cx='24' cy='27' r='2.2' fill='#311B92'/><circle cx='40' cy='27' r='2.2' fill='#311B92'/><rect x='6' y='22' width='6' height='12' rx='2' fill='#5E35B1'/><rect x='52' y='22' width='6' height='12' rx='2' fill='#5E35B1'/><rect x='18' y='44' width='28' height='16' rx='4' fill='#9575CD'/>",
    "network": _net(),
    "share": _share(),
    "stack": "<rect x='10' y='42' width='44' height='12' rx='3' fill='#EF6C00'/><rect x='10' y='27' width='44' height='12' rx='3' fill='#FB8C00'/><rect x='10' y='12' width='44' height='12' rx='3' fill='#FFB74D'/>",
    "gear": "<circle cx='32' cy='32' r='21' fill='none' stroke='#43A047' stroke-width='9' stroke-dasharray='8.2 8.3'/><circle cx='32' cy='32' r='17' fill='#43A047'/><circle cx='32' cy='32' r='7' fill='#fff'/>",
    "receipt": "<path d='M14 6h36v52l-6-4-6 4-6-4-6 4-6-4-6 4z' fill='#fff' stroke='#1E6FC2' stroke-width='3' stroke-linejoin='round'/><path d='M21 18h22M21 27h22M21 36h14' stroke='#1E6FC2' stroke-width='3'/><circle cx='44' cy='44' r='9' fill='#43A047'/><path d='M40 44l3 3 5-6' stroke='#fff' stroke-width='2.5' fill='none'/>",
    "chat": "<rect x='3' y='6' width='38' height='26' rx='7' fill='#43A047'/><path d='M11 32l-2 9 10-9z' fill='#43A047'/><rect x='22' y='24' width='39' height='25' rx='7' fill='#1E88E5'/><path d='M51 49l4 9-12-9z' fill='#1E88E5'/><g fill='#fff'><circle cx='33' cy='37' r='2.6'/><circle cx='41.5' cy='37' r='2.6'/><circle cx='50' cy='37' r='2.6'/></g>",
    "sparkle": "<path d='M30 6l6 19 19 7-19 7-6 19-6-19-19-7 19-7z' fill='#7E57C2'/><path d='M52 4l2 6 6 2-6 2-2 6-2-6-6-2 6-2z' fill='#FFB300'/>",
    "server": "<rect x='10' y='6' width='44' height='15' rx='3' fill='#455A64'/><rect x='10' y='25' width='44' height='15' rx='3' fill='#546E7A'/><rect x='10' y='44' width='44' height='15' rx='3' fill='#607D8B'/><g fill='#69F0AE'><circle cx='18' cy='13.5' r='2.5'/><circle cx='18' cy='32.5' r='2.5'/><circle cx='18' cy='51.5' r='2.5'/></g><path d='M30 13.5h18M30 32.5h18M30 51.5h18' stroke='#CFD8DC' stroke-width='2'/>",
    "chip": "<g stroke='#78909C' stroke-width='3'><path d='M24 6v10M32 6v10M40 6v10M24 48v10M32 48v10M40 48v10M6 24h10M6 32h10M6 40h10M48 24h10M48 32h10M48 40h10'/></g><rect x='15' y='15' width='34' height='34' rx='4' fill='#37474F'/><rect x='23' y='23' width='18' height='18' rx='2' fill='#4FC3F7'/>",
    "monitor": "<rect x='4' y='7' width='56' height='39' rx='4' fill='#263238'/><rect x='8' y='11' width='48' height='31' fill='#E3F2FD'/><rect x='14' y='29' width='6' height='10' fill='#1E88E5'/><rect x='24' y='21' width='6' height='18' fill='#43A047'/><rect x='34' y='25' width='6' height='14' fill='#FB8C00'/><rect x='44' y='17' width='6' height='22' fill='#E53935'/><rect x='26' y='46' width='12' height='8' fill='#455A64'/><rect x='18' y='54' width='28' height='4' rx='2' fill='#455A64'/>",
    "plug": "<path d='M24 5v13M40 5v13' stroke='#546E7A' stroke-width='5' stroke-linecap='round'/><path d='M15 18h34v10a17 17 0 0 1-34 0z' fill='#00ACC1'/><rect x='29' y='44' width='6' height='15' rx='2' fill='#546E7A'/>",
    "coins": "<ellipse cx='24' cy='50' rx='16' ry='6' fill='#F9A825'/><rect x='8' y='30' width='32' height='20' fill='#F9A825'/><ellipse cx='24' cy='40' rx='16' ry='6' fill='#FBC02D' opacity='.6'/><ellipse cx='24' cy='30' rx='16' ry='6' fill='#FFEB3B'/><circle cx='46' cy='22' r='14' fill='#E53935'/><path d='M46 14v10M46 28v2' stroke='#fff' stroke-width='4' stroke-linecap='round'/>",
    "target": "<circle cx='30' cy='34' r='24' fill='#E53935'/><circle cx='30' cy='34' r='17' fill='#fff'/><circle cx='30' cy='34' r='10' fill='#E53935'/><circle cx='30' cy='34' r='4' fill='#fff'/><path d='M30 34 56 8' stroke='#37474F' stroke-width='3'/><path d='M50 6l8-2-2 8z' fill='#37474F'/>",
    "save": "<path d='M8 8h40l8 8v40H8z' fill='#1E6FC2'/><rect x='17' y='8' width='26' height='16' fill='#E3F2FD'/><rect x='34' y='11' width='5' height='10' fill='#1E6FC2'/><rect x='16' y='34' width='32' height='22' fill='#fff'/><path d='M21 41h22M21 48h16' stroke='#90CAF9' stroke-width='2.5'/>",
    "wrench": "<path d='M44 6a14 14 0 0 0-13 19L8 48a5 5 0 0 0 8 8l23-23a14 14 0 0 0 19-13l-8 8-8-2-2-8z' fill='#607D8B'/>",
    "doc": "<path d='M12 4h28l12 12v44H12z' fill='#FB8C00'/><path d='M40 4v12h12z' fill='#FFCC80'/><path d='M19 28h26M19 36h26M19 44h18' stroke='#fff' stroke-width='3'/>",
    "loop": "<path d='M50 24A20 20 0 0 0 13 28' stroke='#43A047' stroke-width='6' fill='none'/><path d='M44 14l12 8-14 6z' fill='#43A047'/><path d='M14 40a20 20 0 0 0 37-4' stroke='#1E88E5' stroke-width='6' fill='none'/><path d='M20 50 8 42l14-6z' fill='#1E88E5'/>",
}


def uri(name):
    svg = f"<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 64 64'>{SVG[name]}</svg>"
    return "data:image/svg+xml," + base64.b64encode(svg.encode()).decode()
