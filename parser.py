if __name__ == '__main__':

    total_items = \
     {'spc p', 'hi-lo', 'oml', 'attac', 'sierra', 'refg', 'isuzu', 'polic', 'dot #', 'a-one', 'pch', 'omr', 'range',
      'chart', 'g omr', 'lumbe', 'stak', 'dual', 'box', 'internatio', 'skidsteerl', 'backhoe', 'rgs', 'tugge', 'priva',
      'rented boo', 'bulld', 'cont', 'hdc', 'aspha', 'liver', 'tcm', 'mcy b', 'ref g', 'hi lo',
      'freightlin', 'dot r', 'intl', 'con e', 'lma', 'rood', 'pallet jac', 'compa', 'lawn', 'vam', 'post', 'mta', 'fre',
      'winne', 'cont-', 'qbe i', 'vespa', 'workm', 'vendor cha', 'com', 'scom', 'message si', 'club', 'omt', 'fllet', 'fd la', 'aport', 'mta u', 'jlg m', 'deagr', 'skid', 'tir', 'g psd', 'btm',
      'fltrl', 'suret', 'conti', 'power ladd', 'trlpm', 'itas', 'yw po'}

    vehicles = set()
    for item in total_items:
        if 'saf' in item:
            vehicles.add(item)

    for item in vehicles:
        total_items.remove(item)

    print(total_items)
    print('=================')
    print(vehicles)