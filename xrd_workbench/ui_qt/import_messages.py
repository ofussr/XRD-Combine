"""Report partially readable measurement files without discarding good ranges."""

from ..localization import localised
from PySide6.QtWidgets import QMessageBox


def import_error_text(error):
    cause = error if getattr(error, 'code', None) else error.__cause__
    if getattr(cause, 'code', None) == 'source_changed':
        name = cause.context['source_name']
        return localised(
            f"Source {name} changed during import. Retry after the file has finished updating.",
            f"La source {name} a changé pendant l’import. Réessayez une fois sa mise à jour terminée.",
            f"Файл {name} изменился во время импорта. Повторите импорт после завершения его записи.",
        )
    return str(error)


def import_warning_messages(documents):
    grouped = {}
    for document in documents:
        metadata = getattr(document.payload, 'metadata', {})
        issues = metadata.get('import_issues', ())
        if issues:
            grouped.setdefault(document.source.name, set()).update(item['index'] for item in issues)
    messages = []
    for name, indices in grouped.items():
        numbers = ', '.join(map(str, sorted(indices)[:12]))
        if len(indices) > 12:
            numbers += ', …'
        messages.append(localised(
            f"{name}: skipped unreadable ranges: {numbers}.",
            f"{name} : plages illisibles ignorées : {numbers}.",
            f"{name}: пропущены нечитаемые диапазоны: {numbers}.",
        ))
    return messages


def show_import_warnings(parent, documents):
    messages = import_warning_messages(documents)
    if messages:
        QMessageBox.warning(parent, localised('Import warning', 'Avertissement d’import', 'Предупреждение при импорте'),
                            '\n'.join(messages[:12]))
