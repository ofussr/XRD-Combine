"""Runtime localisation for the Tk interface."""

from __future__ import annotations

import json
import tkinter as tk
from pathlib import Path
from tkinter import filedialog as _filedialog
from tkinter import messagebox as _messagebox
from tkinter import ttk


LANGUAGES = {"en": "English", "fr": "Français", "ru": "Русский"}
DEFAULT_LANGUAGE = "en"
_language = DEFAULT_LANGUAGE
_settings_path = Path.home() / ".xrd_combine.json"
_legacy_settings_path = Path.home() / ".xrd_workbench.json"


_ENTRIES: list[tuple[str, str, str]] = [
    ('Вид вдоль нормали к (h k l)', 'View along (h k l) normal', 'Vue selon la normale (h k l)'),
    ('Абсолютный поворот', 'Absolute rotation', 'Rotation absolue'),
    ('Относительный поворот', 'Relative rotation', 'Rotation relative'),
    ('Установить углы', 'Set angles', 'Appliquer les angles'),
    ('Повернуть на Δ', 'Rotate by Δ', 'Tourner de Δ'),
    ('Выбранная точка / отражение', 'Selected point / reflection', 'Point / réflexion sélectionné(e)'),
    ('Выберите точку или отражение на графике.', 'Select a point or reflection on the plot.', 'Sélectionnez un point ou une réflexion sur le graphique.'),
    ('Выбранное отражение недоступно при текущих параметрах.', 'The selected reflection is unavailable with the current settings.', 'La réflexion sélectionnée est indisponible avec les paramètres actuels.'),
    ('Ориентация и вращение', 'Orientation and rotation', 'Orientation et rotation'),
    ('Ориентация по (h k l)', 'Orientation by (h k l)', 'Orientation selon (h k l)'),
    ('Смотреть вдоль нормали к (h k l)', 'View along the normal to (h k l)', 'Vue suivant la normale à (h k l)'),
    ('Стандартная ориентация', 'Standard orientation', 'Orientation standard'),
    ('Сбросить поворот', 'Reset rotation', 'Réinitialiser la rotation'),
    ('Базисные векторы', 'Basis vectors', 'Vecteurs de base'),
    ("Координаты атомов не заданы", "No atom positions are available", "Aucune position atomique n’est disponible"),
    ("Структура не загружена", "No structure is loaded", "Aucune structure n’est chargée"),
    ("Выше", "Move up", "Monter"),
    ("Ниже", "Move down", "Descendre"),
    ("Структура CIF…", "CIF structure…", "Structure CIF…"),
    ("Высота CIF, %", "CIF height, %", "Hauteur CIF, %"),
    ("Режим CIF", "CIF mode", "Mode CIF"),
    ("Отдельно", "Separate", "Séparé"),
    ("Наложение", "Overlay", "Superposition"),
    (
        "Наложение CIF доступно только для измерений по оси 2θ.",
        "CIF overlay is available only for measurements on the 2θ axis.",
        "La superposition CIF est disponible uniquement pour les mesures sur l’axe 2θ.",
    ),
    ("Отдельная ось CIF, 2θ", "Separate CIF axis, 2θ", "Axe CIF séparé, 2θ"),
    ("CIF: отдельная ось 2θ", "CIF: separate 2θ axis", "CIF : axe 2θ séparé"),
    ("XY: ось X принята за 2θ; при необходимости выберите другую.",
     "XY: X is assumed to be 2θ; select another axis if needed.",
     "XY : X est supposé être 2θ ; choisissez un autre axe si nécessaire."),
    ("Высота CIF должна быть числом от 10 до 85%.",
     "CIF height must be a number from 10 to 85%.",
     "La hauteur CIF doit être un nombre de 10 à 85 %."),
    ("Для CIF нужны конечные числовые границы: min < max.",
     "CIF limits must be finite numbers: min < max.",
     "Les limites CIF doivent être des nombres finis : min < max."),
    ("Диапазон CIF должен пересекаться с 0–180°.",
     "The CIF range must overlap 0–180°.",
     "La plage CIF doit recouper 0–180°."),
    ("Связи показаны приближённо, по ковалентным радиусам.",
     "Bonds are approximate, based on covalent radii.",
     "Les liaisons sont approximatives, selon les rayons covalents."),
    ("Файл", "File", "Fichier"),
    ("Правка", "Edit", "Édition"),
    (
        "Опорные пики для коррекции…",
        "Reference peaks for correction…",
        "Pics de référence pour la correction…",
    ),
    ("Раздел", "Section", "Section"),
    ("Просмотр", "Viewer", "Visualisation"),
    ("Структуры", "Structures", "Structures"),
    ("Графики", "Plots", "Graphiques"),
    ("Просмотр структуры", "Structure viewer", "Visualisation de la structure"),
    ("Расчётный график", "Calculated graph", "Graphique calculé"),
    ("Отражения", "Reflections", "Réflexions"),
    ("Данные проекта", "Project data", "Données du projet"),
    ("Измерения", "Measurements", "Mesures"),
    ("Структуры CIF", "CIF structures", "Structures CIF"),
    ("Фазы по параметрам ячейки", "Cell-parameter phases", "Phases définies par la maille"),
    ("Фаза по параметрам ячейки", "Cell-parameter phase", "Phase définie par la maille"),
    ("Новая фаза по ячейке…", "New cell phase…", "Nouvelle phase de maille…"),
    ("Редактировать параметры…", "Edit parameters…", "Modifier les paramètres…"),
    ("Пространственная группа", "Space group", "Groupe d’espace"),
    (
        "Введите номер, символ или Hall N.",
        "Enter a number, symbol, or Hall N.",
        "Saisissez un numéro, un symbole ou Hall N.",
    ),
    ("Ячейка", "Cell", "Maille"),
    ("Данные полюсных фигур", "Pole-figure data", "Données des figures de pôles"),
    ("В разделе", "In section", "Dans la section"),
    ("Удалить из проекта", "Remove from project", "Supprimer du projet"),
    ("Полюсная", "Pole figure", "Figure de pôles"),
    ("Справка", "Help", "Aide"),
    ("Язык", "Language", "Langue"),
    ("Цвета атомов", "Atom colours", "Couleurs des atomes"),
    ("Импортировать свои цвета…", "Import custom colours…", "Importer des couleurs personnalisées…"),
    ("Экспортировать свои цвета…", "Export custom colours…", "Exporter les couleurs personnalisées…"),
    ("Сбросить свои цвета", "Reset custom colours", "Réinitialiser les couleurs personnalisées"),
    ("Открыть измерения…", "Open measurements…", "Ouvrir des mesures…"),
    ("Добавить CIF…", "Add CIF…", "Ajouter un CIF…"),
    ("Выход", "Exit", "Quitter"),
    ("О программе", "About", "À propos"),
    ("Коррекция", "Correction", "Correction"),
    ("Полюсные фигуры", "Pole figures", "Figures de pôles"),
    ("Экспериментальная RAW", "Experimental RAW", "Expérimentale RAW"),
    ("Расчётная", "Calculated", "Calculée"),
    ("Данные", "Data", "Données"),
    (
        "Откройте XRDML, RAW, XY или CIF.",
        "Open an XRDML, RAW, XY or CIF file.",
        "Ouvrez un fichier XRDML, RAW, XY ou CIF.",
    ),
    ("Излучение для CIF", "Radiation for CIF", "Rayonnement pour le CIF"),
    ("Загруженные наборы", "Loaded datasets", "Jeux de données chargés"),
    ("Название", "Name", "Nom"),
    ("Вид.", "Vis.", "Vis."),
    ("Тип", "Type", "Type"),
    ("Цвет", "Colour", "Couleur"),
    ("Скрыть", "Hide", "Masquer"),
    ("Показать", "Show", "Afficher"),
    ("Удалить", "Remove", "Supprimer"),
    ("Все", "All", "Tous"),
    ("Очистить", "Clear", "Effacer"),
    ("Переименовать…", "Rename…", "Renommer…"),
    ("Отправить в просмотр", "Send to viewer", "Envoyer vers la visualisation"),
    ("Отправить в коррекцию", "Send to correction", "Envoyer vers la correction"),
    ("Отправить в сравнение", "Send to comparison", "Envoyer vers la comparaison"),
    (
        "Коррекция выбранного измерения",
        "Selected measurement correction",
        "Correction de la mesure sélectionnée",
    ),
    (
        "Выберите измерение в списке.",
        "Select a measurement in the list.",
        "Sélectionnez une mesure dans la liste.",
    ),
    ("Сдвиг X", "X shift", "Décalage X"),
    ("Сдвиг нуля Y", "Y zero shift", "Décalage du zéro Y"),
    ("Масштаб Y", "Y multiplier", "Multiplicateur Y"),
    ("Коррекция по пику…", "Peak correction…", "Correction par pic…"),
    ("Опорные пики…", "Reference peaks…", "Pics de référence…"),
    (
        "Сбросить преобразования",
        "Reset transformations",
        "Réinitialiser les transformations",
    ),
    ("Применить результат", "Apply result", "Appliquer le résultat"),
    ("Добавить новый", "Add new", "Ajouter un nouveau"),
    ("Заменить исходный", "Replace source", "Remplacer la source"),
    ("Сохранить в файл", "Save to file", "Enregistrer dans un fichier"),
    (
        "Сохранить результат…",
        "Save result…",
        "Enregistrer le résultat…",
    ),
    ("Таблица отражений…", "Reflection table…", "Table des réflexions…"),
    ("Таблица отражений", "Reflection table", "Table des réflexions"),
    (
        "Расчётная полюсная фигура…",
        "Calculated pole figure…",
        "Figure de pôles calculée…",
    ),
    ("Отображение", "Display", "Affichage"),
    ("Ось X", "X axis", "Axe X"),
    ("Координата скана", "Scan coordinate", "Coordonnée du balayage"),
    ("Линейная", "Linear", "Linéaire"),
    ("Логарифмическая", "Logarithmic", "Logarithmique"),
    ("Квадратный корень", "Square root", "Racine carrée"),
    ("Квадрат", "Square", "Carré"),
    ("Штрихи", "Sticks", "Raies"),
    ("Профиль", "Profile", "Profil"),
    ("Шкала Y", "Y scale", "Échelle Y"),
    ("Сдвиг кривых", "Curve offset", "Décalage des courbes"),
    ("Фазы", "Phases", "Phases"),
    ("Перестроить", "Redraw", "Redessiner"),
    ("Границы графика", "Plot limits", "Limites du graphique"),
    ("Применить", "Apply", "Appliquer"),
    ("Авто", "Auto", "Auto"),
    ("Сравнение с подложками", "Substrate comparison", "Comparaison aux substrats"),
    ("Рабочее поле", "Workspace", "Espace de travail"),
    ("Галерея и результаты", "Gallery & results", "Galerie et résultats"),
    ("Сбросить расположение", "Reset layout", "Réinitialiser la disposition"),
    (
        "Подложки – правый щелчок создаёт копию",
        "Substrates – right-click to clone",
        "Substrats – clic droit pour dupliquer",
    ),
    (
        "Измерения – перетащите для объединения",
        "Measurements – drag to connect",
        "Mesures – glissez pour regrouper",
    ),
    ("Загрузить настройки…", "Load presets…", "Charger les réglages…"),
    ("Сохранить настройки…", "Save presets…", "Enregistrer les réglages…"),
    ("Обновить галерею", "Update gallery", "Actualiser la galerie"),
    (
        "На рабочем поле нет активных групп.",
        "No active groups exist in the workspace.",
        "Aucun groupe actif dans l’espace de travail.",
    ),
    ("Открыть", "Open", "Ouvrir"),
    ("Копировать", "Copy", "Copier"),
    ("Режим Y", "Y mode", "Mode Y"),
    ("Сохранить и закрыть", "Save & close", "Enregistrer et fermer"),
    ("Подложки – опционально", "Substrates – optional", "Substrats – facultatif"),
    (
        "Перетащите измерение на подложку",
        "Drag either list onto the other",
        "Glissez une liste sur l’autre",
    ),
    (
        "Загрузить папку подложек…",
        "Load substrate folder…",
        "Charger le dossier des substrats…",
    ),
    ("Открыть файлы…", "Open files…", "Ouvrir des fichiers…"),
    ("Открыть папку…", "Open folder…", "Ouvrir un dossier…"),
    ("Линия", "Line", "Raie"),
    ("Излучение", "Radiation", "Rayonnement"),
    ("Своё излучение", "Custom radiation", "Rayonnement personnalisé"),
    (
        "Введите от одной до пяти спектральных линий.",
        "Enter between one and five spectral lines.",
        "Saisissez entre une et cinq raies spectrales.",
    ),
    ("Относительный вес", "Relative weight", "Poids relatif"),
    ("Добавить линию", "Add line", "Ajouter une raie"),
    ("Удалить последнюю", "Remove last", "Supprimer la dernière"),
    ("Отмена", "Cancel", "Annuler"),
    ("Некорректное излучение", "Invalid radiation", "Rayonnement incorrect"),
    ("Вес", "Weight", "Poids"),
    ("линия", "line", "raie"),
    ("вес", "weight", "poids"),
    ("Кратность", "Multiplicity", "Multiplicité"),
    ("кратность", "multiplicity", "multiplicité"),
    ("Iотн, %", "Irel, %", "Irel, %"),
    ("измерение", "measurement", "mesure"),
    ("подложка", "substrate", "substrat"),
    ("Сохранить CSV…", "Save CSV…", "Enregistrer le CSV…"),
    ("Сохранить таблицу", "Save table", "Enregistrer la table"),
    ("Строк:", "Rows:", "Lignes :"),
    ("Переименовать набор", "Rename dataset", "Renommer le jeu de données"),
    ("Новое название:", "New name:", "Nouveau nom :"),
    ("Открыть рентгенограммы", "Open diffraction patterns", "Ouvrir des diffractogrammes"),
    ("Поддерживаемые файлы", "Supported files", "Fichiers pris en charge"),
    ("Текстовые данные", "Text data", "Données texte"),
    ("Открыть папку с измерениями", "Open measurement folder", "Ouvrir le dossier de mesures"),
    ("Добавить CIF", "Add CIF", "Ajouter un CIF"),
    ("Папка с подложками", "Substrate folder", "Dossier des substrats"),
    ("1. Открыть XRDML", "1. Open XRDML", "1. Ouvrir un XRDML"),
    ("1. Открыть данные", "1. Open data", "1. Ouvrir des données"),
    ("Данные XRD", "XRD data", "Données XRD"),
    ("Bruker RAW", "Bruker RAW", "Bruker RAW"),
    ("Данные XY", "XY data", "Données XY"),
    ("Файлы XRDML", "XRDML files", "Fichiers XRDML"),
    ("Файлы XY", "XY Data files", "Fichiers XY"),
    ("Текстовые файлы", "Text files", "Fichiers texte"),
    ("Сдвиг 2θ, °:", "2θ shift, °:", "Décalage 2θ, ° :"),
    ("Сдвиг оси X, °:", "X-axis shift, °:", "Décalage de l’axe X, ° :"),
    ("2а. Сохранить XRDML", "2a. Save XRDML", "2a. Enregistrer le XRDML"),
    ("2б. Сохранить XY", "2b. Save XY", "2b. Enregistrer le XY"),
    ("Вернуть в 2θ", "Send back to 2θ", "Renvoyer vers 2θ"),
    ("Создать копию", "Duplicate", "Dupliquer"),
    ("Аппроксимация", "Peak fit", "Ajustement du pic"),
    ("Выделите область…", "Select an area…", "Sélectionnez une zone…"),
    ("Сдвигать Omega на 1/2", "Shift Omega by 1/2", "Décaler Omega de 1/2"),
    ("Опорные пики", "Reference peaks", "Pics de référence"),
    (
        "Увеличьте пик, нажмите «Аппроксимация» и выделите полезные точки.",
        "Zoom into a peak, press “Peak fit”, then select the useful points.",
        "Agrandissez un pic, cliquez sur « Ajustement du pic », puis sélectionnez les points utiles.",
    ),
    ("База опорных пиков", "Reference peak database", "Base des pics de référence"),
    ("Подложка / название пика", "Substrate / peak name", "Substrat / nom du pic"),
    ("Название:", "Name:", "Nom :"),
    ("Добавить", "Add", "Ajouter"),
    ("Удалить выбранное", "Remove selected", "Supprimer la sélection"),
    ("Сохранить", "Save", "Enregistrer"),
    ("Align Peak", "Align peak", "Aligner le pic"),
    ("Select reference from DB:", "Select a database reference:", "Choisissez une référence :"),
    ("Or enter TRUE 2Theta value:", "Or enter the true 2θ value:", "Ou saisissez la valeur 2θ réelle :"),
    ("OK", "OK", "OK"),
    ("Cancel", "Cancel", "Annuler"),
    ("Экспериментальная полюсная фигура", "Experimental pole figure", "Figure de pôles expérimentale"),
    ("RAW не выбран", "No RAW selected", "Aucun RAW sélectionné"),
    ("Выберите RAW-файл.", "Select a RAW file.", "Sélectionnez un fichier RAW."),
    (
        "Щёлкните по фигуре для чтения точки.",
        "Click the figure to inspect a point.",
        "Cliquez sur la figure pour examiner un point.",
    ),
    ("Открыть RAW…", "Open RAW…", "Ouvrir un RAW…"),
    ("Открыть RAW", "Open RAW", "Ouvrir un RAW"),
    ("Ряд углов наклона", "Tilt-angle series", "Série d’angles d’inclinaison"),
    ("Первый угол, °", "First angle, °", "Premier angle, °"),
    ("Шаг, °", "Step, °", "Pas, °"),
    ("Последний угол, °", "Last angle, °", "Dernier angle, °"),
    (
        "Углы из RAW заполняются автоматически. Для резервных XY задайте любые два значения – третье будет вычислено.",
        "RAW angles are filled automatically. For XY fallback, enter any two values; the third is calculated.",
        "Les angles RAW sont renseignés automatiquement. Pour les fichiers XY de secours, saisissez deux valeurs ; la troisième sera calculée.",
    ),
    ("Цветовая шкала", "Colour scale", "Échelle de couleurs"),
    ("Сплошная заливка", "Solid fill", "Remplissage uni"),
    ("Цвет заливки…", "Fill colour…", "Couleur de remplissage…"),
    ("Шкала интенсивности", "Intensity scale", "Échelle d’intensité"),
    ("Лог.", "Log", "Log"),
    ("Границы интенсивности", "Intensity limits", "Limites d’intensité"),
    ("Нижняя", "Lower", "Inférieure"),
    ("Верхняя", "Upper", "Supérieure"),
    ("Выбранная точка", "Selected point", "Point sélectionné"),
    (
        "Колесо мыши масштабирует фигуру вокруг курсора.",
        "Use the mouse wheel to zoom around the cursor.",
        "Utilisez la molette pour zoomer autour du curseur.",
    ),
    ("Сбросить масштаб", "Reset zoom", "Réinitialiser le zoom"),
    ("Сохранить рисунок…", "Save figure…", "Enregistrer la figure…"),
    ("Выбрать цвет заливки", "Select fill colour", "Choisir la couleur de remplissage"),
    ("Сохранить полюсную фигуру", "Save pole figure", "Enregistrer la figure de pôles"),
    ("Теоретическая полюсная фигура по CIF", "Calculated pole figure from CIF", "Figure de pôles calculée depuis un CIF"),
    ("CIF не открыт", "No CIF loaded", "Aucun CIF chargé"),
    ("Структура CIF", "CIF structure", "Structure CIF"),
    ("Открыть CIF…", "Open CIF…", "Ouvrir un CIF…"),
    ("Центрирование по полюсу", "Pole centring", "Centrage sur un pôle"),
    ("Выбрать разрешённый полюс:", "Select an allowed pole:", "Choisir un pôle autorisé :"),
    (
        "Выберите полюс, не запрещённый систематически:",
        "Select a pole not systematically forbidden:",
        "Choisir un pôle non interdit systématiquement :",
    ),
    ("Поместить полюс (hkl) в центр", "Place pole (h k l) at centre", "Placer le pôle (h k l) au centre"),
    ("Макс. индекс списка", "Maximum list index", "Indice maximal de la liste"),
    ("Отображаемые отражения", "Displayed reflections", "Réflexions affichées"),
    ("d от, Å", "d from, Å", "d min., Å"),
    ("d до, Å", "d to, Å", "d max., Å"),
    (
        "Построить все разрешённые отражения",
        "Plot all allowed reflections",
        "Tracer toutes les réflexions autorisées",
    ),
    (
        "Построить отражения, не запрещённые систематически",
        "Plot reflections not systematically forbidden",
        "Tracer les réflexions non interdites systématiquement",
    ),
    ("Проекция:", "Projection:", "Projection :"),
    ("Стереографическая", "Stereographic", "Stéréographique"),
    ("Равноплощадная", "Equal-area", "Équivalente"),
    ("Подписывать полюса", "Label poles", "Étiqueter les pôles"),
    ("Показать структуру рядом", "Show structure alongside", "Afficher la structure à côté"),
    ("Покрасить точки по d", "Colour points by d", "Colorer les points selon d"),
    ("Размер точек по d", "Point size by d", "Taille des points selon d"),
    ("Цвет точек", "Point colour", "Couleur des points"),
    ("Один цвет", "Uniform colour", "Couleur uniforme"),
    ("Цвет точек по d", "Point colour by d", "Couleur des points selon d"),
    (
        "Цвет точек по расчётной интенсивности",
        "Point colour by calculated intensity",
        "Couleur des points selon l’intensité calculée",
    ),
    (
        "Научные данные и сторонние компоненты",
        "Scientific data and third-party components",
        "Données scientifiques et composants tiers",
    ),
    ("Полные уведомления…", "Full notices…", "Mentions complètes…"),
    ("Закрыть", "Close", "Fermer"),
    ("Расчёт", "Calculation", "Calcul"),
    ("2θ от", "2θ from", "2θ de"),
    ("до", "to", "à"),
    ("Iотн ≥, %", "Irel ≥, %", "Irel ≥, %"),
    ("Рассчитать", "Calculate", "Calculer"),
    (
        "Абсолютный поворот кристалла",
        "Absolute crystal rotation",
        "Rotation absolue du cristal",
    ),
    (
        "Установить абсолютные углы",
        "Set absolute angles",
        "Définir les angles absolus",
    ),
    (
        "Относительный поворот кристалла",
        "Relative crystal rotation",
        "Rotation relative du cristal",
    ),
    (
        "Повернуть относительно текущего",
        "Rotate relative to current",
        "Tourner par rapport à l’orientation actuelle",
    ),
    ("Вернуть центрирующий полюс", "Restore centred pole", "Restaurer le pôle centré"),
    (
        "Совместить выбранный полюс",
        "Align selected pole",
        "Aligner le pôle sélectionné",
    ),
    (
        "Перетаскивание внутри круга свободно вращает кристалл.\n"
        "Щелчок по полюсу выводит его данные справа.",
        "Dragging inside the circle freely rotates the crystal.\n"
        "Clicking a pole shows its data on the right.",
        "Le glissement dans le cercle fait tourner librement le cristal.\n"
        "Un clic sur un pôle affiche ses données à droite.",
    ),
    ("Выбранный полюс", "Selected pole", "Pôle sélectionné"),
    ("Выберите структурный файл CIF", "Select a CIF structure file", "Sélectionnez un fichier de structure CIF"),
    ("Не удалось открыть CIF", "Could not open CIF", "Impossible d’ouvrir le CIF"),
    ("Не удалось открыть файл", "Could not open file", "Impossible d’ouvrir le fichier"),
    ("Ошибка", "Error", "Erreur"),
    ("Ошибка CIF", "CIF error", "Erreur CIF"),
    (
        "Расчёт CIF доступен только для оси 2θ.",
        "CIF calculation is available only for the 2θ axis.",
        "Le calcul CIF n’est disponible que pour l’axe 2θ.",
    ),
    ("Ошибка расчёта", "Calculation error", "Erreur de calcul"),
    ("Ошибка сохранения", "Save error", "Erreur d’enregistrement"),
    ("Готово", "Done", "Terminé"),
    ("Все файлы", "All files", "Tous les fichiers"),
    ("Ошибка разбора", "Parsing Error", "Erreur d’analyse"),
    ("Ошибка аппроксимации", "Fit Error", "Erreur d’ajustement"),
    ("Ошибка экспорта", "Export Error", "Erreur d’exportation"),
    ("Ошибка сохранения", "Save Error", "Erreur d’enregistrement"),
    ("Ошибка базы данных", "Database Error", "Erreur de base de données"),
    ("Отсутствует зависимость", "Missing dependency", "Dépendance manquante"),
    ("Некорректное значение 2θ", "Invalid 2Theta value", "Valeur 2θ incorrecte"),
    ("Сначала загрузите данные.", "Load XRDML file first!", "Chargez d’abord des données."),
    ("Данные не загружены.", "No data loaded!", "Aucune donnée chargée."),
    ("XRDML не загружен.", "No XRDML file loaded!", "Aucun fichier XRDML chargé."),
    (
        "Введите корректное значение сдвига.",
        "Please enter a valid number for the shift!",
        "Saisissez une valeur de décalage valide.",
    ),
    ("Введите корректное число.", "Please enter a valid number!", "Saisissez un nombre valide."),
    (
        "В выбранной области слишком мало точек.",
        "Too few data points inside the selected rectangle.",
        "Trop peu de points dans le rectangle sélectionné.",
    ),
    (
        "В выбранной области нет положительной интенсивности.",
        "Selected region has no positive intensity.",
        "La zone sélectionnée ne contient aucune intensité positive.",
    ),
    (
        "Для гауссовой аппроксимации требуется SciPy.\nУстановите его командой:\npip install scipy",
        "scipy is required for Gaussian fitting.\nInstall it with:\npip install scipy",
        "SciPy est requis pour l’ajustement gaussien.\nInstallez-le avec :\npip install scipy",
    ),
    ("Файл XY сохранён.", "The XY file was saved.", "Le fichier XY a été enregistré."),
    (
        "Исправленный XRDML сохранён.",
        "The corrected XRDML was saved.",
        "Le fichier XRDML corrigé a été enregistré.",
    ),
    ("Ошибка чтения измерения", "Measurement read error", "Erreur de lecture de la mesure"),
    (
        "В выбранном измерении нет конечных значений интенсивности.",
        "The selected measurement contains no finite intensity values.",
        "La mesure sélectionnée ne contient aucune intensité finie.",
    ),
    (
        "Логарифмическая шкала недоступна",
        "Logarithmic scale unavailable",
        "Échelle logarithmique indisponible",
    ),
    (
        "В данных нет положительной интенсивности.",
        "The data contain no positive intensities.",
        "Les données ne contiennent aucune intensité positive.",
    ),
    (
        "Некорректный диапазон отражений",
        "Invalid reflection range",
        "Intervalle de réflexions incorrect",
    ),
    ("Некорректные hkl", "Invalid h k l", "h k l incorrects"),
    ("Некорректный угол", "Invalid angle", "Angle incorrect"),
    (
        "RAW и резервные XY не прочитаны.",
        "Neither RAW nor fallback XY data could be read.",
        "Ni les données RAW ni les données XY de secours n’ont pu être lues.",
    ),
    (
        "В измерении нет конечных значений интенсивности.",
        "The measurement contains no finite intensity values.",
        "La mesure ne contient aucune valeur d’intensité finie.",
    ),
    (
        "Углы наклона не могут быть отрицательными.",
        "Tilt angles cannot be negative.",
        "Les angles d’inclinaison ne peuvent pas être négatifs.",
    ),
    (
        "Для нескольких диапазонов шаг угла не может быть нулевым.",
        "The angle step cannot be zero for multiple ranges.",
        "Le pas angulaire ne peut pas être nul pour plusieurs plages.",
    ),
    (
        "Нижняя граница должна быть меньше верхней.",
        "The lower limit must be below the upper limit.",
        "La limite inférieure doit être inférieure à la limite supérieure.",
    ),
    (
        "Для логарифмической шкалы нижняя граница должна быть положительной.",
        "The lower limit must be positive for a logarithmic scale.",
        "La limite inférieure doit être positive pour une échelle logarithmique.",
    ),
    (
        "Не найдено данных, пригодных для построения.",
        "No plottable data were found.",
        "Aucune donnée exploitable n’a été trouvée.",
    ),
    ("переменный", "variable", "variable"),
    (
        "Углы X, Y и Z должны быть числами.",
        "X, Y and Z angles must be numeric.",
        "Les angles X, Y et Z doivent être numériques.",
    ),
]


_ALIASES: dict[str, tuple[str, str, str]] = {}
for ru, en, fr in _ENTRIES:
    values = (en, fr, ru)
    for alias in values:
        _ALIASES[alias] = values


def get_language() -> str:
    return _language


def load_language() -> str:
    for path in (_settings_path, _legacy_settings_path):
        try:
            value = json.loads(path.read_text(encoding="utf-8")).get("language")
        except (OSError, ValueError, TypeError):
            continue
        if value in LANGUAGES:
            return value
    return DEFAULT_LANGUAGE


def set_language(language: str, persist: bool = False) -> None:
    global _language
    if language not in LANGUAGES:
        language = DEFAULT_LANGUAGE
    _language = language
    if persist:
        try:
            _settings_path.write_text(
                json.dumps({"language": language}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except OSError:
            pass


def translate_text(value: object, language: str | None = None) -> object:
    if not isinstance(value, str):
        return value
    target = language or _language
    translated = _ALIASES.get(value)
    if translated is None:
        return value
    return translated[{"en": 0, "fr": 1, "ru": 2}[target]]


def localised(en: str, fr: str, ru: str) -> str:
    return {"en": en, "fr": fr, "ru": ru}[_language]


def choice_code(group: str, value: str) -> str:
    choices = {
        "scale": {
            "linear": ("Линейная", "Linear", "Linéaire"),
            "log": ("Логарифмическая", "Logarithmic", "Logarithmique"),
            "sqrt": ("Квадратный корень", "Square root", "Racine carrée"),
            "square": ("Квадрат", "Square", "Carré"),
        },
        "phase": {
            "sticks": ("Штрихи", "Sticks", "Raies"),
            "profile": ("Профиль", "Profile", "Profil"),
        },
        "phase_layout": {
            "separate": ("Отдельно", "Separate", "Séparé"),
            "overlay": ("Наложение", "Overlay", "Superposition"),
        },
        "projection": {
            "stereographic": ("Стереографическая", "Stereographic", "Stéréographique"),
            "equal_area": ("Равноплощадная", "Equal-area", "Équivalente"),
        },
    }
    for code, aliases in choices[group].items():
        if value in aliases:
            return code
    return value


class LocalizedStringVar(tk.StringVar):
    def set(self, value) -> None:
        super().set(translate_text(value))


def _translate_widget(widget: tk.Misc, language: str) -> None:
    if hasattr(widget, "localize_heading"):
        widget.localize_heading(language)
    try:
        text = widget.cget("text")
        translated = translate_text(text, language)
        if translated != text:
            widget.configure(text=translated)
    except (tk.TclError, AttributeError):
        pass

    try:
        variable_name = widget.cget("textvariable")
        if variable_name:
            value = widget.getvar(variable_name)
            translated = translate_text(value, language)
            if translated != value:
                widget.setvar(variable_name, translated)
    except (tk.TclError, AttributeError):
        pass

    if isinstance(widget, ttk.Combobox):
        values = tuple(widget.cget("values"))
        translated = tuple(translate_text(value, language) for value in values)
        if translated != values:
            widget.configure(values=translated)

    if isinstance(widget, ttk.Notebook):
        for tab_id in widget.tabs():
            text = widget.tab(tab_id, "text")
            widget.tab(tab_id, text=translate_text(text, language))

    if isinstance(widget, ttk.Treeview):
        for column in ("#0", *widget.cget("columns")):
            try:
                text = widget.heading(column, "text")
                widget.heading(column, text=translate_text(text, language))
            except tk.TclError:
                pass

    if isinstance(widget, tk.Menu):
        end = widget.index("end")
        if end is not None:
            for index in range(end + 1):
                try:
                    label = widget.entrycget(index, "label")
                    widget.entryconfigure(index, label=translate_text(label, language))
                except tk.TclError:
                    pass


def apply_language(root: tk.Misc, language: str | None = None) -> None:
    target = language or _language
    _translate_widget(root, target)
    if isinstance(root, (tk.Tk, tk.Toplevel)):
        try:
            menu_name = root.cget("menu")
            if menu_name:
                apply_language(root.nametowidget(menu_name), target)
        except (tk.TclError, KeyError):
            pass
    for child in root.winfo_children():
        apply_language(child, target)


class _MessageboxProxy:
    def __getattr__(self, name):
        function = getattr(_messagebox, name)

        def call(title=None, message=None, **kwargs):
            if title is not None:
                title = translate_text(title)
            if message is not None:
                message = translate_text(message)
            return function(title=title, message=message, **kwargs)

        return call


class _FileDialogProxy:
    def __getattr__(self, name):
        function = getattr(_filedialog, name)

        def call(**kwargs):
            if "title" in kwargs:
                kwargs["title"] = translate_text(kwargs["title"])
            if "filetypes" in kwargs:
                kwargs["filetypes"] = tuple(
                    (translate_text(label), pattern)
                    for label, pattern in kwargs["filetypes"]
                )
            return function(**kwargs)

        return call


messagebox = _MessageboxProxy()
filedialog = _FileDialogProxy()
