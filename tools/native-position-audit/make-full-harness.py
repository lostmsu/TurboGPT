from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
source = ROOT / 'tools/native-position-audit/original-baseline.cu'
text = source.read_text()
text = text.replace('bool adapters, bool export_hidden=false)', 'bool adapters, bool export_hidden=false, int seen=0)')
text = text.replace('arena.position=p.persistent;\n    int sequences', 'if(!seen)seen=context;\n    arena.position=p.persistent;\n    int sequences')
text = text.replace('tokens[size_t(b)*(context+1)+context]', 'tokens[size_t(b)*(context+1)+seen]')
text = text.replace('size_t(context-1)*Hidden', 'size_t(seen-1)*Hidden')
text = text.replace('Arena arena(12*GiB)', 'Arena arena(20*GiB)')
text = text.replace('runs/qwen-hn1g/original-near256', 'runs/qwen-hn1g/original-near256-fullcontext')
text = text.replace('int effective=job%3==0?16:job%3==1?64:256;', 'int effective=job%3==0?16:job%3==1?64:256;\n            int physical_size=job%3==0?4:job%3==1?8:32,seen=job%3==2?256:258;')
text = text.replace('context!=(job%3==2?256:258)', 'context!=(job%3==0?4096:job%3==1?1024:256)')
text = text.replace('lane+=32', 'lane+=physical_size')
text = text.replace('std::min(32,effective-lane)', 'std::min(physical_size,effective-lane)')
text = text.replace('bytes,false,audit)', 'bytes,false,audit,seen)')
text = text.replace('bytes,true,true)', 'bytes,true,true,seen)')
text = text.replace('std::abs(full.last_nll-expected)', 'std::abs((context==256?full.last_nll:full.position_nll[5])-expected)')
text = text.replace('full.last_bytes!=result.bytes', '(context==256 && full.last_bytes!=result.bytes)')
text = text.replace("<<','<<context<<','<<total_nll", "<<','<<seen<<','<<total_nll")
(ROOT / 'tools/native-position-audit/original-full-baseline.cu').write_text(text, encoding='utf-8')
build = (ROOT / 'tools/native-position-audit/build-baseline.ps1').read_text().replace('original-baseline', 'original-full-baseline')
(ROOT / 'tools/native-position-audit/build-full-baseline.ps1').write_text(build, encoding='utf-8')
