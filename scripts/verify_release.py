#!/usr/bin/env python3
"""
Kiểm một bản kê khai bài nộp (`docs/releases/*.yaml`) — file zip trên đĩa có còn đúng là file
đã nộp không, và nó được sinh ra từ cái gì. CHỦ SỞ HỮU: P3.

VÌ SAO CẦN. `outputs/` và `data/` đều nằm trong `.gitignore`, nên không có gì trong repo ràng
bài nộp vào kho chunk, vào tập câu hỏi, hay vào commit đã sinh ra nó. `plan.md` §1 đòi mọi
submission thật sinh từ một Git tag; tag chứng minh được TRẠNG THÁI MÃ NGUỒN, không chứng minh
được ĐẦU VÀO hay ĐẦU RA. File kê khai lo phần còn lại, script này kiểm nó.

Sáu nhóm kiểm:
  1. tag trỏ đúng commit đã khai (và tag có tồn tại không);
  2. sha256 của mọi đầu vào — bắt ngay việc P2 sinh lại chunks.jsonl mà không ai báo;
  3. sha256 của mọi artefact trung gian và của file zip;
  4. sha256 của file BÊN TRONG zip — zip mang dấu thời gian nên hash ngoài không tái lập được
     giữa hai lần chạy, hash trong thì có, và đó mới là thứ BTC chấm;
  5. cấu trúc submission: đúng tập qid, đúng số doc/câu, không câu rỗng, không quá 5;
  6. các mục `van_de` còn treo — in ra để không ai quên.

    python scripts/verify_release.py --manifest docs/releases/v0.6_private.yaml
    python scripts/verify_release.py --manifest docs/releases/v0.6_private.yaml --run A
    python scripts/verify_release.py --manifest docs/releases/v0.6_private.yaml --commands
    python scripts/verify_release.py --manifest docs/releases/v0.6_private.yaml --record

`--record` ghi lại sha256 THỰC TẾ vào file kê khai. Dùng sau khi chạy lại chuỗi lệnh, hoặc khi
lập kê khai cho một lượt nộp mới — KHÔNG dùng để "sửa cho hết đỏ".
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import yaml  # noqa: E402

CHUNK = 1 << 20


def sha256_file(path: Path) -> tuple[str, int]:
    h, n = hashlib.sha256(), 0
    with path.open("rb") as fh:
        while block := fh.read(CHUNK):
            h.update(block)
            n += len(block)
    return h.hexdigest(), n


def git(*args: str) -> str | None:
    try:
        return subprocess.check_output(["git", *args], cwd=REPO, stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None


class Report:
    """Gom kết quả rồi in một lần. Fail loud, nhưng in HẾT chứ không dừng ở lỗi đầu tiên —
    biết cả năm chỗ hỏng trong một lượt chạy thì sửa được một lần."""

    def __init__(self) -> None:
        self.ok: list[str] = []
        self.warn: list[str] = []
        self.fail: list[str] = []

    def add(self, good: bool, msg: str, soft: bool = False) -> bool:
        (self.ok if good else (self.warn if soft else self.fail)).append(msg)
        return good

    def show(self) -> int:
        for m in self.ok:
            print(f"  ✅ {m}")
        for m in self.warn:
            print(f"  ⚠️  {m}")
        for m in self.fail:
            print(f"  ❌ {m}")
        print(f"\n{len(self.ok)} đạt · {len(self.warn)} cảnh báo · {len(self.fail)} hỏng")
        return 1 if self.fail else 0


def check_hashes(entries: dict, rep: Report, label: str, record: bool) -> None:
    """`entries` = {đường dẫn: {sha256, bytes, inner?}}. `record=True` thì ghi đè giá trị đo được."""
    for rel, want in (entries or {}).items():
        p = REPO / rel
        if not p.exists():
            rep.add(False, f"[{label}] thiếu {rel}")
            continue
        got, size = sha256_file(p)
        if record:
            want["sha256"], want["bytes"] = got, size
            rep.add(True, f"[{label}] ghi lại {rel} → {got[:16]}…")
        else:
            rep.add(
                got == want.get("sha256"),
                f"[{label}] {rel}"
                + ("" if got == want.get("sha256") else
                   f" — sha256 LỆCH\n         khai: {want.get('sha256')}\n         thực: {got}"
                   f"\n         ({want.get('bytes')} vs {size} byte)"),
            )
        if "inner" in want:
            check_zip_inner(p, want["inner"], rep, label, record)


def check_zip_inner(path: Path, inner: dict, rep: Report, label: str, record: bool) -> None:
    with zipfile.ZipFile(path) as z:
        have = {n for n in z.namelist() if not n.endswith("/")}
        for name, want in inner.items():
            if name not in have:
                rep.add(False, f"[{label}] {path.name} không chứa {name} (có: {sorted(have)})")
                continue
            data = z.read(name)
            got = hashlib.sha256(data).hexdigest()
            if record:
                want["sha256"], want["bytes"] = got, len(data)
                rep.add(True, f"[{label}] ghi lại {path.name}::{name} → {got[:16]}…")
            else:
                rep.add(
                    got == want.get("sha256"),
                    f"[{label}] {path.name}::{name}"
                    + ("" if got == want.get("sha256") else
                       f" — sha256 LỆCH (nội dung nộp ĐÃ ĐỔI, không phải chỉ dấu thời gian zip)"),
                )


def check_submission_shape(path: Path, expect: dict, questions: Path | None, rep: Report, label: str) -> None:
    """
    Bốn bất biến của mã chấm BTC (AGENTS.md §3), kiểm lại trên chính file đã nộp.

    Không thay `scripts/p4_check_submission.py` (kiểm TRƯỚC khi nộp, có cả chế độ chấm bằng mã
    BTC) — cái này kiểm SAU, trên file kê khai, và chạy được cả khi không còn corpus.
    """
    with zipfile.ZipFile(path) as z:
        names = [n for n in z.namelist() if not n.endswith("/")]
        if names != ["submission.json"]:
            rep.add(False, f"[{label}] {path.name} chứa {names}, phải đúng ['submission.json']")
            return
        sub = json.loads(z.read("submission.json"))

    n_want = expect.get("n_questions")
    if n_want is not None:
        rep.add(len(sub) == n_want, f"[{label}] {len(sub)} câu (khai {n_want})")

    sizes = {len(v.get("answer", [])) if isinstance(v, dict) else -1 for v in sub.values()}
    if -1 in sizes:
        rep.add(False, f"[{label}] có câu thiếu key 'answer' → mã chấm BTC CRASH, CodaLab báo Failed")
        return
    rep.add(max(sizes) <= 5, f"[{label}] số doc mỗi câu ∈ {sorted(sizes)} (trần cứng 5)")
    rep.add(min(sizes) > 0, f"[{label}] không có câu nào trả list rỗng")
    d_want = expect.get("docs_per_question")
    if d_want is not None:
        rep.add(sizes == {d_want}, f"[{label}] mọi câu đúng {d_want} doc" if sizes == {d_want}
                else f"[{label}] khai {d_want} doc/câu, thực tế {sorted(sizes)}")

    bad_type = [q for q, v in sub.items() if any(not isinstance(d, str) for d in v["answer"])]
    rep.add(not bad_type, f"[{label}] mọi doc_id là str"
            if not bad_type else
            f"[{label}] {len(bad_type)} câu có doc_id KHÔNG phải str (vd {bad_type[:3]}) — "
            f"mã chấm dùng set(a)&set(b) nên 100 != '100': 0 điểm IM LẶNG")
    dup = [q for q, v in sub.items() if len(set(v["answer"])) != len(v["answer"])]
    rep.add(not dup, f"[{label}] không câu nào trùng doc_id"
            if not dup else
            f"[{label}] {len(dup)} câu có doc_id trùng (vd {dup[:3]}) — mẫu số Precision là "
            f"len(list) chứ không phải len(set)")

    if questions and questions.exists():
        qids = {str(q) for q in json.loads(questions.read_text(encoding="utf-8"))}
        miss, extra = qids - set(sub), set(sub) - qids
        rep.add(not miss and not extra,
                f"[{label}] tập qid khớp {questions.name}" if not miss and not extra else
                f"[{label}] thiếu {len(miss)} / thừa {len(extra)} qid so với {questions.name} "
                f"→ mã chấm CRASH và có thể vẫn tiêu một lượt nộp")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--run", default=None, help="chỉ kiểm một lượt (vd A). Mặc định: tất cả.")
    ap.add_argument("--commands", action="store_true", help="in chuỗi lệnh tái lập rồi thoát")
    ap.add_argument("--record", action="store_true", help="GHI ĐÈ sha256 trong kê khai bằng giá trị đo được")
    ap.add_argument("--skip-inputs", action="store_true", help="bỏ qua hash data/ (chunks.jsonl 576 MB)")
    a = ap.parse_args()

    mpath = REPO / a.manifest
    man = yaml.safe_load(mpath.read_text(encoding="utf-8"))
    runs = {k: v for k, v in man["runs"].items() if a.run is None or k == a.run}
    if not runs:
        raise SystemExit(f"❌ kê khai không có lượt '{a.run}'. Có: {', '.join(man['runs'])}")

    if a.commands:
        for name, run in runs.items():
            print(f"\n# ── lượt {name} — {run.get('mo_ta','')} ──")
            for s in man.get("shared_steps", []) + run.get("steps", []):
                print(f"\n# {s['id']} → {s['out']}" + (f"   ({s['chi_phi']})" if s.get("chi_phi") else ""))
                print(s["cmd"])
        return 0

    print(f"kê khai : {a.manifest}\nbản      : {man['release']} ({man.get('ngay')})\n"
          f"lượt     : {', '.join(runs)}\nchế độ   : {'GHI LẠI' if a.record else 'kiểm'}\n")
    rep = Report()

    # 1. đầu vào
    if a.skip_inputs:
        rep.add(True, "[input] bỏ qua theo --skip-inputs", soft=True)
    else:
        check_hashes(man.get("inputs", {}), rep, "input", a.record)

    # 2. từng lượt
    for name, run in runs.items():
        tag, commit = run.get("tag"), run.get("commit")
        if tag:
            at = git("rev-list", "-n", "1", tag)
            if at is None:
                rep.add(False, f"[{name}] tag '{tag}' không tồn tại trong repo này")
            elif commit:
                rep.add(at == commit, f"[{name}] tag '{tag}' → {at[:12]}"
                        + ("" if at == commit else f", kê khai ghi {commit[:12]}"))
        else:
            rep.add(False, f"[{name}] KHÔNG có git tag. plan.md §1: mọi submission thật phải "
                           f"sinh từ một tag.", soft=True)

        lb = run.get("leaderboard") or {}
        if lb.get("recall") is None:
            rep.add(False, f"[{name}] chưa có điểm leaderboard trong kê khai", soft=True)
        else:
            rep.add(True, f"[{name}] LB recall={lb['recall']} precision={lb.get('precision')}")

        check_hashes(run.get("artifacts", {}), rep, name, a.record)

        # 3. cấu trúc file nộp
        expect = man.get("expect", {})
        qpath = None
        for rel in man.get("inputs", {}):
            if "official" in rel:
                qpath = REPO / rel
        for rel in run.get("artifacts", {}):
            if rel.endswith(".zip") and (REPO / rel).exists():
                check_submission_shape(REPO / rel, expect, qpath, rep, name)

    code = rep.show()

    if a.record:
        mpath.write_text(yaml.safe_dump(man, allow_unicode=True, sort_keys=False), encoding="utf-8")
        print(f"\n✅ đã ghi lại {a.manifest}")
        print("⚠️  yaml.safe_dump KHÔNG giữ comment — xem `git diff` trước khi commit, và khôi "
              "phục phần ghi chú nếu nó biến mất.")

    treo = man.get("van_de") or []
    if treo:
        print(f"\n── {len(treo)} vấn đề còn treo trong kê khai ──")
        for v in treo:
            print(f"  [{v.get('muc','?'):8s}] {v['id']}: {' '.join(v['mo_ta'].split())[:150]}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
