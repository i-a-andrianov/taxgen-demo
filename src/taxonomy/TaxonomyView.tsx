import { useEffect, useRef, useState } from "react";
import { DataSet } from "vis-data";
import { Network } from "vis-network/standalone/esm/vis-network";
import Button from 'react-bootstrap/Button';
import Card from 'react-bootstrap/Card';
import Row from 'react-bootstrap/Row';
import Col from 'react-bootstrap/Col';
import Form from 'react-bootstrap/Form';
import Table from 'react-bootstrap/Table';
import InputGroup from 'react-bootstrap/InputGroup';
import { Taxonomy, Word } from "./TaxonomyDTO";

export interface TaxonomyViewProps {
  taxonomy: Taxonomy;
  misc: Word[];
  navigateToRoot: () => void;
  navigateToWord: (id: string) => void;
  navigateToSearch: (search: string) => void;
  generateWords: (id: string) => void;
  generateRelations: (fromId: string, toId: string) => void;
  regenerateGraph: (id: string) => void;
}

export default function TaxonomyView(props: TaxonomyViewProps) {
    const {
      taxonomy,
      misc,
      navigateToRoot, 
      navigateToWord, 
      navigateToSearch,
      generateWords,
      generateRelations,
      regenerateGraph
    } = props;
    const {currentWord, words, relations} = taxonomy;
    const definition = words.filter((w) => w.id === currentWord).map((w) => w.definition)[0];
    const lemmas = words.filter((w) => w.id === currentWord).map((w) => w.lemmas)[0];
    const [flag, setFlag] = useState<string | null>(null);
    const ref = useRef<HTMLDivElement>(null);

    useEffect(() => {
      if (!ref.current) return;
  
      const nodes = new DataSet<any>(
        words.map((w) => {
          return {
            id: w.id,
            label: w.word,
            shape: "circularImage", 
            image: `/api/images/${w.word}`,
            color: {background: (w.id === currentWord) ? '#8cffdd' : (w.generated ? '#9effff' : '#ccd1ff'), 
                    border: (w.id === currentWord) ? '#8cffdd' : '#ccd1ff',
                    hover: {border: "#f50041", background: (w.id === currentWord) ? '#8cffdd' : (w.generated ? '#9effff' : '#ccd1ff')},
                    highlight: {background: (w.id === currentWord) ? '#8cffdd' : (w.generated ? '#9effff' : '#ccd1ff')}
                  },
            level: w.level
          };
        })
      );
  
      const edges = new DataSet<any>(
        relations.map((r) => {
           return {
             id: r.parent + '$$$' + r.child,
             from: r.parent,
             to: r.child
            };
        })
      );
  
      const data = {
        nodes: nodes,
        edges: edges,
      };

      const options = {
        nodes: {
          borderWidth: 3,
          borderWidthSelected: 5
        },
        edges: {
            arrows: 'to',
            color: '#3642b3'
        },
        height: '800px',
        width: '100%',
        layout: {
          hierarchical: {
            enabled: true,
            direction: 'UD',
            sortMethod: 'directed'
          }
        },
        physics: {
          "barnesHut": {
            "springConstant": 0,
            "avoidOverlap": 0.5
          }
        },
        interaction: {
          dragNodes: false,
          hover: true,
          navigationButtons: true,
          keyboard: true
        },
        clickToUse: false
      };
      
      const network = new Network(ref.current, data, options);

      network.on('doubleClick', (e) => {
        const id = e.nodes[0];
        if (!id) {
          const eid = e.edges[0];
          if (eid) {
            const [fromId, toId] = eid.split('$$$');
            generateRelations(fromId, toId);
          }
          return;
        }
        if (id !== currentWord) {
          navigateToWord(id);
        } else {
          generateWords(id);
        }
      });

      network.on("afterDrawing", function(ctx) {
        var srcCanvas = ctx.canvas
        var destinationCanvas = document.createElement("canvas");
        destinationCanvas.width = srcCanvas.width;
        destinationCanvas.height = srcCanvas.height;

        var destCtx = destinationCanvas.getContext('2d')!;

        //create a rectangle with the desired color
        destCtx.fillStyle = "#FFFFFF";
        destCtx.fillRect(0,0,srcCanvas.width,srcCanvas.height);

        //draw the original canvas onto the destination canvas
        destCtx.drawImage(srcCanvas, 0, 0);

        //finally use the destinationCanvas.toDataURL() method to get the desired output;
        destinationCanvas.toDataURL();
        
        const element = document.getElementById('canvasImg') as HTMLLinkElement;
        element.href = destinationCanvas.toDataURL();
      })
  
      return () => {
        network.off('hold');
        network.off('doubleClick');
        network.destroy();
      }
    }, [currentWord, words, relations, navigateToWord, generateWords, generateRelations]);

    useEffect(() => {
      if (!currentWord) { setFlag(null); return; }
      fetch(`/api/images/${encodeURIComponent(currentWord)}`, { method: "HEAD" })
        .then(res => setFlag(res.headers.get("X-Image-Source")))
        .catch(() => setFlag(null));
    }, [currentWord]);

    
  
    const [search, setSearch] = useState('');
    return (<>
      <h2>
        <br/>
      TaxFree: WordNet3.0 visualization for candidate-free taxonomy enrichment
      <br/><br/>
      </h2>
      <Row>
        <Col xs={1}>
          <Button style={{ "backgroundColor": "#008CBA", "borderColor": "#008CBA" } as React.CSSProperties} onClick={navigateToRoot}>Back to root</Button>
        </Col>
        <Col xs={1}>
          <Button style={{ "backgroundColor": "#008CBA", "borderColor": "#008CBA" }as React.CSSProperties} onClick={() => regenerateGraph(currentWord)}>Reset graph</Button>
        </Col>
        <Col xs={3}>
          <InputGroup className="mb-3">
            <Form.Control type="text" placeholder="word" value={search} onChange={(e) => setSearch(e.target.value)} onKeyPress={event => {
              if (event.key === "Enter") {navigateToSearch(search); setSearch("");
              }
            }} />
            <Button style={{"backgroundColor": "#008CBA", "borderColor": "#008CBA", "paddingLeft": "10px !important"} as React.CSSProperties} onClick={() => {navigateToSearch(search); setSearch("")}}>Move to</Button>
          </InputGroup>
        </Col>
        <Col xs={2}>
          <a id="canvasImg" download="schema.png">
            <Button style={{ "backgroundColor": "#008CBA", "borderColor": "#008CBA" }as React.CSSProperties}>Download image</Button>
          </a>
        </Col>
      </Row>
      <Row>
        <Col xs={9}>
          <div ref={ref}/>
        </Col>
        <Col>
          {currentWord ?
            <Card>
              <div style={{ display: "flex", alignItems: "center", justifyContent: "center", gap: 12, paddingTop: 8 }}>
               <Button
                variant="light"
                onClick={() => {/* no-op for now */}}
                aria-label="Previous"
                className="shadow-sm"
                style={{ borderRadius: "50%", width: 40, height: 40, padding: 0, border: "1px solid #222" }}
              >
                <span style={{ fontSize: 20, lineHeight: "40px" }}>‹</span>
              </Button>
              <Card.Img variant="top" src={`/api/images/${currentWord}`}/>
              <Button
                variant="light"
                onClick={() => {/* no-op for now */}}
                aria-label="Next"
                className="shadow-sm"
                style={{ borderRadius: "50%", width: 40, height: 40, padding: 0, border: "1px solid #222" }}
              >
                <span style={{ fontSize: 20, lineHeight: "40px" }}>›</span>
              </Button>
              </div>
              <Card.Body>
                <Card.Text><i>{flag === "generated" ? "AI generated" : "Original image"}</i></Card.Text>
                <Card.Title>{currentWord}</Card.Title>
                <Card.Text>
                  {lemmas.join()}
                </Card.Text>
                <Card.Text>
                  {definition}
                </Card.Text>
              </Card.Body>
            </Card>
            :
            <></>
          }
          {misc.length ?
          <><span><br/>Maybe you meant:</span>
          <Table striped bordered hover size="sm">
            <tbody>
              {misc.slice(1).map(s => 
                  <tr>
                  <td><a href="#" onClick={() => navigateToWord(s.word)}>{s.word}</a></td>
                  <td>{s.definition}</td>
                </tr>
                )} 
            </tbody>
          </Table></> : <></>}
        </Col>
      </Row>
    </>);
}
