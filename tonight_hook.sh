#!/bin/bash
# run_tonight.sh 의 [4.5] 에서 실행 (11/4 밤, 무인)
#   1) white.dae 에서 횡단보도를 찾아 라벨 설정에 추가 + 이미 모은 사진에 횡단보도 라벨 덧붙이기
#   2) Gazebo(B) 가 켜져 있으면 횡단보도 앞 집중 수집 250자세
#   3) 학습·점검이 끝나면 결과(모델, 점검표, 로그, 확인 사진)를 tmp/yolo-1104 브랜치에 올리는 작업을 예약
D=$1; CFG=$2
V=~/yolo_data/v2
S=$(ros2 pkg prefix kudos_yolo_tools)/lib/kudos_yolo_tools
RAW=https://raw.githubusercontent.com/hotsixthekingrush/KUDOS_scalecar/tmp/yolo-1104
count() { ls $D/labels/all 2>/dev/null | wc -l; }

echo "  [hook] 횡단보도 찾기 $(date +%H:%M)"
if curl -fsSL $RAW/crosswalk_from_paint.py -o $V/crosswalk_from_paint.py; then
  W=$(ls -t /tmp/kudos_arena_*/track_2026.world 2>/dev/null | head -1)
  [ -z "$W" ] && W=~/26v2/KUDOS_Autorace2026_Gazebo/src/kudos_arena/worlds/track_2026.world
  echo "  world: $W"
  python3 $V/crosswalk_from_paint.py --world "$W" --mesh-root ~/26v2/KUDOS_Autorace2026_Gazebo \
    --cfg $CFG --data $D --lib $S
  rc=$?
else
  echo "  crosswalk_from_paint.py 못 받음"; rc=9
fi

if [ $rc -eq 0 ] && [ -f $CFG/crosswalks.txt ] && timeout 5 ros2 topic echo --once /gazebo/model_states > /dev/null 2>&1; then
  ONLY=$(cat $CFG/crosswalks.txt | tr -d '\n')
  echo "  [hook] 횡단보도 집중 수집 250자세 ($ONLY) $(date +%H:%M)"
  start=$(count); target=$((start + 240)); prev=-1; still=0
  ros2 launch kudos_yolo_tools collect_data_v2.launch.py config_dir:=$CFG output_dir:=$D \
    num_poses:=250 only:=$ONLY branch_sign:=B > $V/collect_cw.log 2>&1 &
  while true; do
    n=$(count); echo "  $(date +%H:%M) 저장 $n / 목표 $target"
    [ $n -ge $target ] && break
    if [ $n -eq $prev ]; then still=$((still+1)); else still=0; fi
    [ $still -ge 40 ] && { echo "  10분 동안 안 늘어서 그만"; break; }
    prev=$n; sleep 15
  done
  sleep 30
  pkill -INT -f viewpoint_sampler.py; pkill -INT -f auto_labeler.py; sleep 5
  pkill -INT -f collect_data_v2.launch.py; sleep 3
  python3 - "$D" <<'EOF'
import json, os, sys, glob
import numpy as np
d = sys.argv[1]; removed = 0; prev = {}
for m in sorted(glob.glob(os.path.join(d, 'meta', '*.json'))):
    stem = os.path.splitext(os.path.basename(m))[0]
    sess = stem.rsplit('_', 1)[0]
    T = np.array(json.load(open(m))['T_cam_world'])
    if sess in prev and np.allclose(T, prev[sess], atol=1e-4):
        for p in (('images/all', '.jpg'), ('labels/all', '.txt'), ('debug', '.jpg'), ('meta', '.json')):
            f = os.path.join(d, p[0], stem + p[1])
            if os.path.exists(f):
                os.remove(f)
        removed += 1
    else:
        prev[sess] = T
print('  겹친 사진 %d장 삭제' % removed)
EOF
else
  echo "  [hook] 횡단보도 수집 건너뜀 (rc=$rc 또는 Gazebo 없음)"
fi
echo "  클래스별 라벨 수 (0 신호등 1 A 2 B 3 P 4 보행자 6 방지턱 7 LIMO 9 횡단보도):"
cat $D/labels/all/*.txt 2>/dev/null | awk 'NF{print $1}' | sort -n | uniq -c

# --- 학습·점검 끝나면 결과 올리기 (백그라운드) ---
cat > $V/push_results.sh <<'EOF'
#!/bin/bash
V=~/yolo_data/v2; LOG=$V/tonight.log
for i in $(seq 1 480); do
  grep -q -e "===== 끝" -e "학습 실패" $LOG && break
  sleep 30
done
sleep 10
R=~/26contest/src/KUDOS_scalecar
WT=/tmp/yolo_push_1104
export GIT_TERMINAL_PROMPT=0
{
  echo "[push] $(date)"
  git -C $R fetch origin tmp/yolo-1104 && git -C $R worktree add -f --detach $WT origin/tmp/yolo-1104 || exit 1
  O=$WT/results_1104; mkdir -p $O/debug
  cp $V/models/best_sim_full_1104.pt $O/ 2>/dev/null
  cp $(ls -t $V/mission_check_*.csv 2>/dev/null | head -1) $O/ 2>/dev/null
  cp $LOG $O/tonight.log
  cp $V/config_full/*.yaml $V/config_full/*.txt $O/ 2>/dev/null
  RUN=$(ls -td $V/runs/detect/sim_full_1104*/ 2>/dev/null | head -1)
  [ -n "$RUN" ] && cp $RUN/results.csv $RUN/confusion_matrix.png $O/ 2>/dev/null
  ls $V/sim_full/debug_cw/*.jpg 2>/dev/null | head -6 | xargs -r -I{} cp {} $O/debug/
  for c in 4 6 2; do
    for f in $(grep -l "^$c " $V/sim_full/labels/all/*.txt 2>/dev/null | head -2); do
      b=$(basename $f .txt); cp $V/sim_full/debug/$b.jpg $O/debug/c${c}_$b.jpg 2>/dev/null
    done
  done
  cd $WT && git add results_1104 && \
  (git commit -q -m "results: 11/4 밤 시뮬 학습·미션 점검 결과 (동아리방 노트북 자동)" || \
   git -c user.name=cosmosxyz4 -c user.email=cosmosxyz4@gmail.com commit -q -m "results: 11/4 밤 시뮬 학습·미션 점검 결과 (동아리방 노트북 자동)") && \
  git push origin HEAD:tmp/yolo-1104 && echo "[push] 완료" || echo "[push] 실패 (인증?) — 결과는 $V 에 있음"
  cd ~ && git -C $R worktree remove --force $WT
} >> $V/push_results.log 2>&1
EOF
nohup bash $V/push_results.sh > /dev/null 2>&1 &
echo "  [hook] 결과 올리기 예약됨 (로그: $V/push_results.log)"
