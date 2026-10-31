"""Check the actual A5 RegBase dispatch predicate without CANN or an NPU."""
import argparse
from pathlib import Path
import re
import subprocess
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--compiler', default='g++')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    host = root / 'xllm_ops/mega_gdn_mtp_decode/op_host/mega_gdn_mtp_decode_tiling_a5.inc'
    match = re.search(r'const bool use_a5_regbase_deferred\s*=\s*(.*?);',
                      host.read_text(), re.S)
    if match is None:
        raise RuntimeError('A5 RegBase dispatch predicate was not found')
    source = '''
#include <iostream>
struct Info { int batch_size, speculative_tokens, num_k_heads, num_v_heads; };
bool actual(Info info, bool fla, bool is_ascend950, bool use_deferred_norm) {
  bool* fla_ssm_state_layout = &fla;
  return PREDICATE;
}
int main() {
  int cases = 0;
  for (bool fla : {false, true}) for (bool a5 : {false, true})
  for (bool deferred : {false, true}) for (int b = 1; b <= 8; ++b)
  for (int s = 0; s <= 5; ++s) for (int nk : {8, 16})
  for (int nv : {16, 24, 48}) {
    const bool expected = fla && a5 && deferred && s == 3 && nk == 16 &&
        ((b == 4 && (nv == 16 || nv == 48)) || (b == 5 && nv == 48));
    if (actual({b, s, nk, nv}, fla, a5, deferred) != expected) return 1;
    ++cases;
  }
  std::cout << cases << " dispatch cases passed\\n";
}
'''.replace('PREDICATE', match.group(1))
    with tempfile.TemporaryDirectory(prefix='mtp-a5-dispatch-') as tmp:
        cpp = Path(tmp) / 'dispatch.cpp'
        binary = Path(tmp) / 'dispatch'
        cpp.write_text(source)
        subprocess.run([args.compiler, '-std=c++17', '-O2', str(cpp),
                        '-o', str(binary)], check=True)
        subprocess.run([str(binary)], check=True)


if __name__ == '__main__':
    main()
