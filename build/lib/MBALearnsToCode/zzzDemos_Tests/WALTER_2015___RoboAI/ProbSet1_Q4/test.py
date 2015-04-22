import pip
from pprint import pprint
installed_packages = pip.get_installed_distributions()
installed_packages_list = sorted(["%s==%s" % (i.key, i.version)
     for i in installed_packages])
pprint(installed_packages_list)


from frozen_dict import FrozenDict

fd = FrozenDict(a =1)
print(fd)

import MBALearnsToCode.
