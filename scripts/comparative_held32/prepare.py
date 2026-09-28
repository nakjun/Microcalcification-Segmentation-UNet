from common import prepare
if __name__=='__main__':
 c=prepare();print(f"prepared held={len(c['rows'])} checkpoints={len(c['checkpoints'])} signature={c['signature']}")
