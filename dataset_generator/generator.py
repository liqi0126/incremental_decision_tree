import subprocess
import sys
import getopt
import os

configs = [
    "java -cp moa.jar -javaagent:sizeofag-1.0.4.jar moa.DoTask \"WriteStreamToARFFFile -s (generators.RandomTreeGenerator -r 1 -i 1 -c 2 -o 5 -u 0 -v 5 -d 5 -l 3 -f 0.15) -f ../../tmp.arff -m {0}\"",
    "java -cp moa.jar -javaagent:sizeofag-1.0.4.jar moa.DoTask \"WriteStreamToARFFFile -s (generators.RandomTreeGenerator -r 1 -i 1 -c 3 -o 5 -u 0 -v 5 -d 5 -l 3 -f 0.15) -f ../../tmp.arff -m {0}\"",
    "java -cp moa.jar -javaagent:sizeofag-1.0.4.jar moa.DoTask \"WriteStreamToARFFFile -s (generators.RandomTreeGenerator -r 1 -i 1 -c 4 -o 5 -u 0 -v 5 -d 5 -l 3 -f 0.15) -f ../../tmp.arff -m {0}\"",
    "java -cp moa.jar -javaagent:sizeofag-1.0.4.jar moa.DoTask \"WriteStreamToARFFFile -s (generators.RandomTreeGenerator -r 1 -i 1 -c 5 -o 5 -u 0 -v 5 -d 5 -l 3 -f 0.15) -f ../../tmp.arff -m {0}\"",
]


def arff2csv(arff_path, csv_path=None, _encoding='utf8'):
    with open(arff_path, 'r', encoding=_encoding) as fr:
        attributes = []
        if csv_path is None:
            csv_path = arff_path[:-4] + 'csv'  # *.arff -> *.csv
        write_sw = False
        with open(csv_path, 'w', encoding=_encoding) as fw:
            for line in fr.readlines():
                if write_sw:
                    fw.write(line.replace('value','').replace('class','')[:-2]+'\n')
                elif '@data' in line:
                    fw.write(','.join(attributes))
                    write_sw = True
                elif '@attribute' in line:
                    # @attribute attribute_tag numeric
                    attributes.append(line.split()[1])
    print("Convert {} to {}.".format(arff_path, csv_path))


if __name__ == '__main__':
    args, _ = getopt.getopt(sys.argv[1:], "n:")
    dataset_size = '5000'

    for o, a in args:
        if o == '-n' and a:
            dataset_size = a

    wd = os.getcwd()
    for i in range(len(configs)):
        subprocess.check_call(configs[i].format(dataset_size), shell=True, cwd=wd + '/moa/lib')
        arff2csv('tmp.arff', 'dataset{0}.csv'.format(str(i + 1)))
    os.remove(wd + '/tmp.arff')
    
