#!/usr/bin/env python3
"""Traduz o FBD do plc.xml (PLCopen) para expressoes ST, para CONFERIR por texto o que cada saida calcula.
Uso: python3 fbd_para_st.py plc.xml"""
import sys, xml.etree.ElementTree as ET
NS = {'p': 'http://www.plcopen.org/xml/tc6_0201'}
def traduz(path):
    r = ET.parse(path).getroot(); fbd = r.find('.//p:pou/p:body/p:LD', NS)
    els = {c.get('localId'): c for c in fbd if c.get('localId')}
    def E(lid, fp=None):
        c = els[str(lid)]; tg = c.tag.split('}')[1]
        if tg == 'inVariable': return c.find('p:expression', NS).text
        if tg == 'block':
            args = []
            for v in c.findall('p:inputVariables/p:variable', NS):
                cn = v.find('p:connectionPointIn/p:connection', NS)
                args.append('%s := %s' % (v.get('formalParameter'), E(cn.get('refLocalId'), cn.get('formalParameter'))) if c.get('instanceName') else E(cn.get('refLocalId'), cn.get('formalParameter')))
            if c.get('instanceName'): return '%s.%s' % (c.get('instanceName'), fp or 'OUT')
            return '%s(%s)' % (c.get('typeName'), ', '.join(args))
        return '?'
    out = {}
    for c in fbd:
        tg = c.tag.split('}')[1]
        if tg == 'outVariable':
            cn = c.find('p:connectionPointIn/p:connection', NS); out[c.find('p:expression', NS).text] = E(cn.get('refLocalId'), cn.get('formalParameter'))
        if tg == 'block' and c.get('instanceName'):
            args = []
            for v in c.findall('p:inputVariables/p:variable', NS):
                cn = v.find('p:connectionPointIn/p:connection', NS); args.append('%s := %s' % (v.get('formalParameter'), E(cn.get('refLocalId'), cn.get('formalParameter'))))
            out['CALL ' + c.get('instanceName')] = '%s(%s)' % (c.get('instanceName'), ', '.join(args))
    return out
if __name__ == '__main__':
    for k, v in sorted(traduz(sys.argv[1]).items()): print('%s := %s;' % (k, v) if not k.startswith('CALL') else v + ';')
