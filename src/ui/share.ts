/**
 * Real share integrations for the review screen (Phase 7): a Web Share
 * button plus a direct Save button on browsers that can share files
 * (mainly mobile Safari/Chrome) — Save is offered alongside Share rather
 * than only via the OS share sheet's own "Save Image" action, since that's
 * an extra tap buried among share targets. Browsers without file-share
 * support get Save + Pinterest + Facebook instead. Split out from
 * reviewScreen.ts for the same reason capture.ts and countdownOverlay.ts
 * are their own files — a distinct browser-API concern, not just DOM
 * layout.
 */

const SHARE_TITLE = 'My Collectibliss AR try-on';

function buildShareFilename(productName: string): string {
  const slug = productName
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/(^-|-$)/g, '');
  return `collectibliss-tryon${slug ? `-${slug}` : ''}.png`;
}

function buildShareText(productName: string): string {
  return productName
    ? `Trying on the ${productName} with Collectibliss AR`
    : 'Trying on jewelry with Collectibliss AR';
}

// Some browsers only partially implement the Web Share API (e.g. `share()`
// for text/url but no file support at all) and a couple of older partial
// implementations throw rather than returning false for an unsupported
// ShareData shape — same defensive try/catch style as tracking.ts's GPU
// delegate fallback.
function supportsFileShare(file: File): boolean {
  try {
    return typeof navigator.canShare === 'function' && navigator.canShare({ files: [file] });
  } catch {
    return false;
  }
}

async function shareFile(file: File, productName: string): Promise<void> {
  try {
    await navigator.share({
      files: [file],
      title: SHARE_TITLE,
      text: buildShareText(productName),
    });
  } catch (error) {
    // Fires when the user dismisses the OS share sheet without picking a
    // target — not a failure, nothing to report.
    if (error instanceof DOMException && error.name === 'AbortError') {
      return;
    }
    console.error('[CollectiblissTryOn] share failed', error);
  }
}

function downloadBlob(blob: Blob, filename: string): void {
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = filename;
  link.style.display = 'none';
  document.body.append(link);
  link.click();
  link.remove();
  // A handful of browsers start the download asynchronously after click()
  // returns; revoking on the next tick avoids a race against that without
  // holding the object URL alive any longer than necessary.
  setTimeout(() => URL.revokeObjectURL(url), 0);
}

// Pinterest's `media` param and Facebook's `u` param are both fetched
// server-side to build the share preview, so both need a publicly
// reachable image URL — a local blob: URL only exists inside this browser
// tab. This widget has no backend to host the snapshot at a real URL
// (CLAUDE.md: everything is client-side), so v1's pragmatic call is:
// point both links at the current product page instead (window.location.href
// is safe to use here since the widget is injected directly into the page
// DOM, never an iframe — CLAUDE.md's iOS Safari note already requires
// that), pulling in that page's own preview image rather than the AR
// snapshot. Paired with a Download button + a hint asking the user to
// attach the downloaded photo manually if they want it on those platforms.
// Revisit if/when a backend or signed-upload endpoint exists to host
// snapshots at a real URL.
function buildPinterestShareUrl(productName: string): string {
  const params = new URLSearchParams({
    url: window.location.href,
    description: buildShareText(productName),
  });
  return `https://www.pinterest.com/pin/create/button/?${params.toString()}`;
}

function buildFacebookShareUrl(): string {
  const params = new URLSearchParams({ u: window.location.href });
  return `https://www.facebook.com/sharer/sharer.php?${params.toString()}`;
}

function openShareWindow(url: string): void {
  window.open(url, '_blank', 'noopener,noreferrer');
}

function createButton(label: string, variant: 'primary' | 'secondary'): HTMLButtonElement {
  const button = document.createElement('button');
  button.type = 'button';
  button.className = `button button--${variant}`;
  button.textContent = label;
  return button;
}

function createSaveButton(
  photoBlob: Blob,
  filename: string,
  variant: 'primary' | 'secondary',
): HTMLButtonElement {
  const button = createButton('Save photo', variant);
  button.addEventListener('click', () => downloadBlob(photoBlob, filename));
  return button;
}

export function createShareSection(photoBlob: Blob, productName: string): HTMLElement {
  const section = document.createElement('div');
  section.className = 'share-section';

  const file = new File([photoBlob], buildShareFilename(productName), { type: 'image/png' });

  if (supportsFileShare(file)) {
    const shareButton = createButton('Share', 'primary');
    shareButton.addEventListener('click', () => {
      shareButton.disabled = true;
      shareFile(file, productName).finally(() => {
        shareButton.disabled = false;
      });
    });
    // A direct save option alongside Share, not just Share alone: the OS
    // share sheet usually has a "Save Image" action buried among its
    // targets, but a one-tap Save button here is faster and doesn't depend
    // on the user knowing it's in there.
    section.append(shareButton, createSaveButton(photoBlob, file.name, 'secondary'));
    return section;
  }

  const pinterestButton = createButton('Share to Pinterest', 'secondary');
  pinterestButton.addEventListener('click', () =>
    openShareWindow(buildPinterestShareUrl(productName)),
  );

  const facebookButton = createButton('Share to Facebook', 'secondary');
  facebookButton.addEventListener('click', () => openShareWindow(buildFacebookShareUrl()));

  const hint = document.createElement('p');
  hint.className = 'share-hint';
  hint.textContent =
    'Save your photo first, then attach it when you post — Pinterest and Facebook can only pull the product page automatically here.';

  section.append(
    createSaveButton(photoBlob, file.name, 'primary'),
    pinterestButton,
    facebookButton,
    hint,
  );
  return section;
}
