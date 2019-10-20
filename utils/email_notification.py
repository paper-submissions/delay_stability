def dict_to_table(dic):
    str = '{:<8} {:<25} {:<10}\n'.format('Number', 'Name', 'Value')
    for num, v in enumerate(sorted(dic.items())):
        label, k = v
        str = str + '{:<8} {:<25} {:}\n'.format(num, label, k)
    return str
