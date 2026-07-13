// 画面右下の Python 環境ステータス表示 + 環境リセットボタン
import { pyEnv, type EnvState } from './py-env';
import { getStrings } from '../strings';

export function mountStatusPill() {
  const t = getStrings();

  const pill = document.createElement('div');
  pill.className = 'py-status-pill';
  pill.hidden = true;

  const label = document.createElement('span');
  label.className = 'py-status-label';

  const resetBtn = document.createElement('button');
  resetBtn.type = 'button';
  resetBtn.className = 'py-status-reset';
  resetBtn.textContent = t.resetEnv;
  resetBtn.title = t.envResetDone;
  resetBtn.hidden = true;
  resetBtn.addEventListener('click', () => pyEnv.reset());

  pill.append(label, resetBtn);
  document.body.appendChild(pill);

  let timer: ReturnType<typeof setInterval> | null = null;

  pyEnv.onState((state: EnvState) => {
    if (timer) {
      clearInterval(timer);
      timer = null;
    }
    pill.hidden = state.kind === 'idle';
    pill.dataset.state = state.kind;
    resetBtn.hidden = !(state.kind === 'ready' || state.kind === 'running' || state.kind === 'error');

    if (state.kind === 'booting') {
      const base = state.phase === 'loading-packages' ? t.bootPackages : t.bootLoading;
      const detail = state.detail ? ` (${state.detail})` : '';
      const update = () => {
        const sec = ((performance.now() - state.startedAt) / 1000).toFixed(1);
        label.textContent = `${base}${detail} … ${sec}${t.seconds}`;
      };
      update();
      timer = setInterval(update, 100);
    } else if (state.kind === 'ready') {
      label.textContent = `✓ ${t.ready}`;
    } else if (state.kind === 'running') {
      label.textContent = t.running;
    } else if (state.kind === 'error') {
      label.textContent = `⚠ ${t.envError}: ${state.message}`;
    }
  });
}
