#!/usr/bin/env python3
"""Gera o projeto OpenPLC corrigido a partir do original (plc.xml em FBD): acrescenta o passa-baixa na VP,
liga a derivada a VP_F e remove o SEL(MF,0.0,...) da derivada.  Uso: python3 gerar_projeto_corrigido.py <plc.xml original> <plc.xml saida>"""
import re, sys
src, dst = sys.argv[1], sys.argv[2]
s = open(src, encoding='utf-8').read()

# 1) derivada passa a usar VP_F (bloco MUL 113 le o inVariable 110)
s, n = re.subn(r'(<inVariable localId="110".*?<expression>)VP_PERC(</expression>)', r'\1VP_F\2', s, count=1, flags=re.S); assert n == 1
# 2) remove SEL(MF, 0.0, .) da derivada (blocos/variaveis 1, 2, 9) e liga a saida direto no MUL 113
for tag, lid in (('block', 1), ('inVariable', 2), ('inVariable', 9)):
    s, n = re.subn(r'\s*<%s localId="%d".*?</%s>' % (tag, lid, tag), '', s, count=1, flags=re.S); assert n == 1
old = '''<connection refLocalId="1" formalParameter="OUT">
                  <position x="910" y="440"/>
                  <position x="890" y="440"/>
                </connection>'''
new = '''<connection refLocalId="113" formalParameter="OUT">
                  <position x="910" y="440"/>
                  <position x="800" y="440"/>
                  <position x="800" y="480"/>
                  <position x="730" y="480"/>
                </connection>'''
assert old in s; s = s.replace(old, new, 1)
# 3) novas variaveis
def var(name, val, doc):
    return '''            <variable name="%s">
              <type>
                <REAL/>
              </type>
              <initialValue>
                <simpleValue value="%s"/>
              </initialValue>
              <documentation>
                <xhtml:p><![CDATA[%s]]></xhtml:p>
              </documentation>
            </variable>
''' % (name, val, doc)
newvars = var('VP_F', '0.0', 'VP FILTRADA (PASSA-BAIXA) PARA A DERIVADA. MALHA ABERTA: VP_F=VP_PERC') + var('kf_VP', '0.017699', 'FILTRO DA VP: kf=Ts/(tauf+Ts)=0.02/(1.11+0.02); tauf=mu*Td=1.11 s')
i = s.index('          </localVars>\n        </interface>'); s = s[:i] + newvars + s[i:]
# 4) nova rede (passa-baixa na VP)
def invar(lid, x, y, w, expr, outy=10):
    return '''            <inVariable localId="%d" height="30" width="%d" negated="false">
              <position x="%d" y="%d"/>
              <connectionPointOut>
                <relPosition x="%d" y="%d"/>
              </connectionPointOut>
              <expression>%s</expression>
            </inVariable>
''' % (lid, w, x, y, w, outy, expr)
def conn(ref, pts, fp=None):
    return '<connection refLocalId="%d"%s>\n' % (ref, '' if fp is None else ' formalParameter="%s"' % fp) + ''.join('                      <position x="%d" y="%d"/>\n' % p for p in pts) + '                    </connection>'
def block(lid, typ, x, y, ins, h=60):
    names = ['IN1', 'IN2', 'IN3'] if typ != 'SEL' else ['G', 'IN0', 'IN1']
    body = ''
    for k, (ref, srcpt, fp) in enumerate(ins):
        ry = 30 + 20 * k; yy = y + ry; sx, sy = srcpt
        pts = [(x, yy), (x - 10, yy), (x - 10, sy), (sx, sy)] if sy != yy else [(x, yy), (sx, sy)]
        body += '''                <variable formalParameter="%s">
                  <connectionPointIn>
                    <relPosition x="0" y="%d"/>
                    %s
                  </connectionPointIn>
                </variable>
''' % (names[k], ry, conn(ref, pts, fp))
    return '''            <block localId="%d" typeName="%s" height="%d" width="70">
              <position x="%d" y="%d"/>
              <inputVariables>
%s              </inputVariables>
              <inOutVariables/>
              <outputVariables>
                <variable formalParameter="OUT">
                  <connectionPointOut>
                    <relPosition x="70" y="30"/>
                  </connectionPointOut>
                </variable>
              </outputVariables>
            </block>
''' % (lid, typ, h, x, y, body)
net = '''            <comment localId="300" height="80" width="700">
              <position x="10" y="2530"/>
              <content>
                <xhtml:p><![CDATA[FILTRO PASSA-BAIXA NA VP ANTES DA DERIVADA (CORRECAO DO RUIDO NO SO)
VP_F = VP_F + kf_VP*(VP_PERC - VP_F), kf_VP = Ts/(tauf+Ts) = 0.02/(1.11+0.02)
MALHA ABERTA: VP_F = VP_PERC (derivada armada, sem pulso ao fechar a malha)]]></xhtml:p>
              </content>
            </comment>
'''
net += invar(301, 10, 2620, 70, 'VP_PERC') + invar(302, 10, 2680, 60, 'VP_F')
net += block(303, 'SUB', 130, 2610, [(301, (80, 2630), None), (302, (70, 2690), None)])
net += invar(304, 130, 2700, 70, 'kf_VP')
net += block(305, 'MUL', 250, 2620, [(303, (200, 2640), 'OUT'), (304, (200, 2710), None)])
net += invar(306, 250, 2710, 60, 'VP_F')
net += block(307, 'ADD', 370, 2630, [(306, (310, 2720), None), (305, (320, 2650), 'OUT')])
net += invar(309, 400, 2600, 30, 'MF') + invar(310, 350, 2740, 70, 'VP_PERC')
net += block(308, 'SEL', 490, 2610, [(309, (430, 2610), None), (310, (420, 2750), None), (307, (440, 2660), 'OUT')], h=80)
net += '''            <outVariable localId="311" height="30" width="60" negated="false">
              <position x="610" y="2620"/>
              <connectionPointIn>
                <relPosition x="0" y="20"/>
                %s
              </connectionPointIn>
              <expression>VP_F</expression>
            </outVariable>
''' % conn(308, [(610, 2640), (560, 2640)], 'OUT')
i = s.rindex('          </LD>'); s = s[:i] + net + s[i:]
open(dst, 'w', encoding='utf-8').write(s)
print('ok ->', dst)
