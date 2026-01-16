from pymol import cmd


def working_analysis():
    """Рабочий анализ без сложных операций"""

    print("АНАЛИЗ МУТАЦИИ")
    print("=" * 50)

    # 1. Загрузка
    cmd.load("orig.cif", "wt")
    cmd.load("mutant.cif", "mut")

    # Запрос позиции мутации
    mut_pos = input("Введите позицию мутации: ")
    try:
        mut_pos = int(mut_pos)  # Преобразуем в число
    except ValueError:
        print("Ошибка: позиция должна быть числом!")
        return

    print(f"\nАнализ мутации в позиции {mut_pos}")

    # Определяем аминокислоты в этой позиции
    wt_aa = get_residue_info("wt", mut_pos)
    mut_aa = get_residue_info("mut", mut_pos)

    print(f"WT: {wt_aa}{mut_pos} → MUT: {mut_aa}{mut_pos}")

    print(f"\nЗагружено: WT={cmd.count_atoms('wt')} атомов, MUT={cmd.count_atoms('mut')} атомов")

    # 2. Выравнивание
    rmsd = cmd.align("mut", "wt")[0]
    print(f"\nВыравнивание: RMSD = {rmsd:.3f} Å")

    print(f"\n1. ОТКЛОНЕНИЯ СТРУКТУРЫ:")

    # 1. Общее
    print(f"   Общее (RMSD): {rmsd:.3f} Å")

    # 2. В окне ±5 остатков
    window_start = mut_pos - 5
    window_end = mut_pos + 6  # +6 потому что range не включает последнее значение
    window_res = list(range(window_start, window_end))
    window_dists = []

    for res in window_res:
        if res > 0:  # Проверяем, чтобы номер остатка был положительным
            if (cmd.count_atoms(f"wt and resi {res} and name CA") and
                    cmd.count_atoms(f"mut and resi {res} and name CA")):
                dist = cmd.distance(f"tmp{res}",
                                    f"wt and resi {res} and name CA",
                                    f"mut and resi {res} and name CA")
                window_dists.append(dist)
                cmd.delete(f"tmp{res}")

    if window_dists:
        avg_win = sum(window_dists) / len(window_dists)
        print(f"   Окно ±5 остатков: {avg_win:.3f} Å (среднее)")

    # 3. Точечное отклонение
    if (cmd.count_atoms(f"wt and resi {mut_pos} and name CA") and
            cmd.count_atoms(f"mut and resi {mut_pos} and name CA")):
        ca_dist = cmd.distance(f"tmp_ca{mut_pos}",
                               f"wt and resi {mut_pos} and name CA",
                               f"mut and resi {mut_pos} and name CA")
        print(f"   Точечное Cα({mut_pos}): {ca_dist:.3f} Å")
        cmd.delete(f"tmp_ca{mut_pos}")

    # 4. Анализ геометрии вокруг мутации
    print(f"\n2. ГЕОМЕТРИЯ ВОКРУГ МУТАЦИИ:")

    # Соседние остатки (предыдущий и следующий)
    prev_res = mut_pos - 1
    next_res = mut_pos + 1

    for neighbor in [prev_res, next_res]:
        if neighbor > 0:  # Проверяем, чтобы номер остатка был положительным
            if (cmd.count_atoms(f"wt and resi {neighbor} and name CA") > 0 and
                    cmd.count_atoms(f"mut and resi {neighbor} and name CA") > 0):
                # WT
                dist_wt = cmd.distance(f"tmp_wt_{mut_pos}_{neighbor}",
                                       f"wt and resi {mut_pos} and name CA",
                                       f"wt and resi {neighbor} and name CA")
                # MUT
                dist_mut = cmd.distance(f"tmp_mut_{mut_pos}_{neighbor}",
                                        f"mut and resi {mut_pos} and name CA",
                                        f"mut and resi {neighbor} and name CA")

                diff = dist_mut - dist_wt

                print(f"   Cα({mut_pos})-Cα({neighbor}):")
                print(f"     WT:  {dist_wt:.3f} Å")
                print(f"     MUT: {dist_mut:.3f} Å")
                print(f"     Δ = {diff:+.3f} Å")

                cmd.delete(f"tmp_wt_{mut_pos}_{neighbor}")
                cmd.delete(f"tmp_mut_{mut_pos}_{neighbor}")

    # 5. Анализ водородных связей
    print(f"\n3. АНАЛИЗ ВОДОРОДНЫХ СВЯЗЕЙ:")

    # Проверяем стандартную водородную связь в белковой цепи
    if prev_res > 0:
        if (cmd.count_atoms(f"wt and resi {prev_res} and name O") > 0 and
                cmd.count_atoms(f"wt and resi {mut_pos} and name N") > 0):

            hb_wt = cmd.distance(f"tmp_hb_wt_{prev_res}_{mut_pos}",
                                 f"wt and resi {prev_res} and name O",
                                 f"wt and resi {mut_pos} and name N")

            print(f"   WT: O({prev_res})-N({mut_pos}) = {hb_wt:.3f} Å", end=" ")

            if hb_wt < 3.5:
                print(f"✓ Возможна H-связь")
            else:
                print(f"✗ Слишком далеко")

            cmd.delete(f"tmp_hb_wt_{prev_res}_{mut_pos}")

    # Универсальная проверка для мутанта
    if mut_aa == "PRO":
        print(f"   MUT: Пролин не имеет амидного водорода!")
        print(f"    Нарушение H-связи в белковой цепи")
    else:
        # Для других аминокислот проверяем сохранение H-связи
        if prev_res > 0:
            if (cmd.count_atoms(f"mut and resi {prev_res} and name O") > 0 and
                    cmd.count_atoms(f"mut and resi {mut_pos} and name N") > 0):

                hb_mut = cmd.distance(f"tmp_hb_mut_{prev_res}_{mut_pos}",
                                      f"mut and resi {prev_res} and name O",
                                      f"mut and resi {mut_pos} and name N")

                print(f"   MUT: O({prev_res})-N({mut_pos}) = {hb_mut:.3f} Å", end=" ")

                if hb_mut < 3.5:
                    print(f"Возможна H-связь")
                else:
                    print(f"Слишком далеко")

                cmd.delete(f"tmp_hb_mut_{prev_res}_{mut_pos}")

    # Вызов новых функций анализа
    analyze_secondary_structure(mut_pos)
    analyze_dihedral_angles(mut_pos, wt_aa, mut_aa)


def get_residue_info(structure, position):
    """Получает информацию об аминокислоте в указанной позиции"""
    info = {}
    cmd.iterate(f"{structure} and resi {position} and name CA",
                "info['resn'] = resn",
                space={'info': info})
    return info.get('resn', '?')


def analyze_secondary_structure(mut_pos):
    """Анализ вторичной структуры вокруг мутации"""

    print(f"\n4. ВЛИЯНИЕ НА ВТОРИЧНУЮ СТРУКТУРУ:")

    # Выполняем DSSP анализ
    cmd.dss("wt")
    cmd.dss("mut")

    # Универсальный анализ водородных связей в окне
    analyze_hydrogen_bonds_window(mut_pos)

    # Геометрический анализ
    analyze_secondary_structure_simple(mut_pos)


def analyze_hydrogen_bonds_window(mut_pos):
    """Анализ водородных связей в окне вокруг мутации"""

    print(f"\n   АНАЛИЗ ВОДОРОДНЫХ СВЯЗЕЙ в окне ±2 остатка:")

    start_res = max(1, mut_pos - 2)
    end_res = mut_pos + 3

    # Проверяем возможные H-связи в окне
    for i in range(start_res, end_res - 1):
        for j in range(i + 1, min(i + 5, end_res + 2)):  # Проверяем соседние остатки
            if (cmd.count_atoms(f"wt and resi {i} and name O") > 0 and
                    cmd.count_atoms(f"wt and resi {j} and name N") > 0):

                dist_wt = cmd.distance(f"tmp_wt_{i}_{j}",
                                       f"wt and resi {i} and name O",
                                       f"wt and resi {j} and name N")

                dist_mut = cmd.distance(f"tmp_mut_{i}_{j}",
                                        f"mut and resi {i} and name O",
                                        f"mut and resi {j} and name N")

                # Выводим только интересные связи (близкие или изменившиеся)
                if dist_wt < 3.5 or dist_mut < 3.5 or abs(dist_mut - dist_wt) > 0.5:
                    print(f"   O({i})-N({j}): WT={dist_wt:.2f}Å → MUT={dist_mut:.2f}Å", end="")

                    if dist_wt < 3.5 and dist_mut >= 3.5:
                        print(f"    ПОТЕРЯНА")
                    elif dist_wt >= 3.5 and dist_mut < 3.5:
                        print(f"    НОВАЯ")
                    elif dist_wt < 3.5 and dist_mut < 3.5:
                        print(f" ✓ СОХРАНЕНА")
                    else:
                        print(f" - нет H-связи")

                cmd.delete(f"tmp_wt_{i}_{j}")
                cmd.delete(f"tmp_mut_{i}_{j}")


def analyze_secondary_structure_simple(mut_pos):
    """Простой анализ вторичной структуры по геометрии"""

    print(f"\n   ГЕОМЕТРИЧЕСКИЙ АНАЛИЗ (окно ±3 остатка):")

    # Анализ расстояний между Cα атомами в окне
    start_res = max(1, mut_pos - 3)
    end_res = mut_pos + 4

    for res1 in range(start_res, end_res - 2):
        res2 = res1 + 2

        if (cmd.count_atoms(f"wt and resi {res1} and name CA") > 0 and
                cmd.count_atoms(f"wt and resi {res2} and name CA") > 0):

            dist_wt = cmd.distance(f"tmp_wt_ca_{res1}_{res2}",
                                   f"wt and resi {res1} and name CA",
                                   f"wt and resi {res2} and name CA")

            dist_mut = cmd.distance(f"tmp_mut_ca_{res1}_{res2}",
                                    f"mut and resi {res1} and name CA",
                                    f"mut and resi {res2} and name CA")

            # Определяем тип структуры по расстоянию
            if 5.5 < dist_wt < 7.5:
                structure_type = "β-слой"
            elif 5.0 < dist_wt < 5.5:
                structure_type = "α-спираль"
            else:
                structure_type = "петля/поворот"

            # Выводим изменение если оно существенное или если это окно мутации
            if abs(dist_mut - dist_wt) > 0.3 or (res1 <= mut_pos <= res2):
                print(f"   Cα({res1})-Cα({res2}): {dist_wt:.1f}Å → {dist_mut:.1f}Å ({structure_type})")
                if abs(dist_mut - dist_wt) > 0.5:
                    print(f"      Значительное изменение геометрии!")

            cmd.delete(f"tmp_wt_ca_{res1}_{res2}")
            cmd.delete(f"tmp_mut_ca_{res1}_{res2}")


def analyze_dihedral_angles(mut_pos, wt_aa, mut_aa):
    """Универсальный анализ торсионных углов для любой мутации"""

    print(f"\n5. ТОРСИОННЫЕ УГЛЫ (φ/ψ) для остатка {mut_pos}:")

    # Определяем соседние остатки
    prev_res = mut_pos - 1
    next_res = mut_pos + 1

    if (cmd.count_atoms(f"wt and resi {mut_pos} and name N") > 0 and
            cmd.count_atoms(f"wt and resi {mut_pos} and name CA") > 0):

        try:
            # φ угол (C(i-1)-N(i)-CA(i)-C(i))
            phi_wt = cmd.get_dihedral(f"wt and resi {prev_res} and name C",
                                      f"wt and resi {mut_pos} and name N",
                                      f"wt and resi {mut_pos} and name CA",
                                      f"wt and resi {mut_pos} and name C")

            phi_mut = cmd.get_dihedral(f"mut and resi {prev_res} and name C",
                                       f"mut and resi {mut_pos} and name N",
                                       f"mut and resi {mut_pos} and name CA",
                                       f"mut and resi {mut_pos} and name C")

            # ψ угол (N(i)-CA(i)-C(i)-N(i+1))
            psi_wt = cmd.get_dihedral(f"wt and resi {mut_pos} and name N",
                                      f"wt and resi {mut_pos} and name CA",
                                      f"wt and resi {mut_pos} and name C",
                                      f"wt and resi {next_res} and name N")

            psi_mut = cmd.get_dihedral(f"mut and resi {mut_pos} and name N",
                                       f"mut and resi {mut_pos} and name CA",
                                       f"mut and resi {mut_pos} and name C",
                                       f"mut and resi {next_res} and name N")

            print(f"   φ угол ({prev_res}C-{mut_pos}N-{mut_pos}CA-{mut_pos}C):")
            print(f"     WT:  {phi_wt:.1f}°")
            print(f"     MUT: {phi_mut:.1f}°")
            print(f"     Δ = {phi_mut - phi_wt:+.1f}°")

            print(f"\n   ψ угол ({mut_pos}N-{mut_pos}CA-{mut_pos}C-{next_res}N):")
            print(f"     WT:  {psi_wt:.1f}°")
            print(f"     MUT: {psi_mut:.1f}°")
            print(f"     Δ = {psi_mut - psi_wt:+.1f}°")

            # Универсальные проверки
            print(f"\n   ОБЩИЙ АНАЛИЗ:")

            # Проверка на стерические затруднения
            if abs(phi_mut) < 30 or abs(psi_mut) < 30:
                print(f"      Возможные стерические затруднения!")

            # Проверка на значительные изменения
            phi_change = abs(phi_mut - phi_wt)
            psi_change = abs(psi_mut - psi_wt)

            if phi_change > 30 or psi_change > 30:
                print(f"      Значительное изменение углов (> 30°)")
            elif phi_change > 15 or psi_change > 15:
                print(f"      Умеренное изменение углов (15-30°)")
            else:
                print(f"      Минимальное изменение углов (< 15°)")

            # Специфичные проверки для разных типов аминокислот

        except Exception as e:
            print(f"   Ошибка при расчете углов: {e}")
            print(f"   Проверьте наличие всех необходимых атомов в остатках {prev_res}, {mut_pos}, {next_res}")


if __name__ == "__main__":
    working_analysis()

