import React, { useState, useEffect, useContext } from "react"
import './savedTest.css'
import { LoaderContext } from './context'
import { Container, Row, Col, Spinner, Modal, Button } from "react-bootstrap";
import {Link} from "react-router-dom";
import Files from 'react-files'
import Downloadpicture from '../images/cloud-computing.png'
import BackArrow from '../images/backArrow.png'
import DeleteIcon from '../images/delete.png'
import axios from "axios";
import LocalStorage from 'local-storage'
import { CSVLink, CSVDownload } from "react-csv";

// import { CgDatabase } from "react-icons/cg";
const modalValues = {
  values: [],
}

const footerValues = {
  total: '',
  correct: '',
  grade: '',
  improvementFeedback: '',
  percentage: '',
  rollNo: ''
}

const correctArr = []
const gradeArr = []
const improvementFeedbackArr = []
const percentageArr = []
const totalArr = []

let headers
let dataing

const SavedTests = () => {

  // const [loaderContext, setLoaderContext] = useContext(LoaderContext)
  const [file, setFile] = useState('')
  const [gradefile, setGradeFile] = useState('')
  const [data, setData] = useState('')
  const [feedback, setFeedback] = useState(false)
  const [feedbackData, setFeedbackData] = useState('')
  const [loading, setLoading] = useState(false)
  const [modal, setModal] = useState(false)
  const [reload, setReload] = useState(false)
  const [number, setNumber] = useState('')
  const [name, setName] = useState('')
  const [improvementFeedback, setImprovementFeedback] = useState('')
  const [errormsg, setErrorMsg] = useState('')
  const user_name = LocalStorage.get('username')

  let renderTestDetails = () => {
    
    if (data.length === 0) {
      return (
        <div className="no_dataDiv">
          {/* <CgDatabase className="noDataIcon" /> */}
          <h4>No Data</h4>
        </div>
      );
    } else if (!feedback) {
      return data.map((test) => {
        console.log('MAPING', test)
        return (
          <Row className="details_row" style={{ borderBottom: '1px solid #e6e6e6' }} > 
            <Col className="test_col">
              <h4 style={{ cursor: 'pointer' }} onClick={() => {
                getFeedback(test.pk)
                setFeedback(true)    
              }}>{test.fields.test_name}</h4>
            </Col>
            <Col className="test_col">
              <h4>{test.fields.test_no}</h4>
            </Col>
            <Col className="test_col">
              <img src={DeleteIcon} style={{ marginTop: 2, height: '24px', width: '24px'}} onClick={() => deleteIcon(test.pk)}/>
            </Col>
          </Row>
        );
      });
    }
    else if (feedback && feedbackData) {
      let Counter = 0 
      return (
        feedbackData.map( data => {
          const { Correct, Grade, Improvement_Feedback, Percentage, Total } = data
          correctArr.push(Correct)
          gradeArr.push(Grade)
          improvementFeedbackArr.push(Improvement_Feedback)
          percentageArr.push(Percentage)
          totalArr.push(Total)
          return (
            <Row className="details_row" onClick={() => {
              modalValues.index = Counter
              setFeedback(true) }}> 
              <Col className="test_col"  >
                <h4 
                  style={{ cursor: 'pointer' }}
                  onClick={() => {
                  footerValues.total = data.Total
                  footerValues.percentage = data.Percentage
                  footerValues.correct = data.Correct
                  footerValues.improvementFeedback = data.Improvement_Feedback
                  footerValues.grade = data.Grade
                  setImprovementFeedback(data.Improvement_Feedback)
                  modalValues['values'] = Object.keys(data).map(a => data[a])
                  setModal(true)}}
                >
                  {data['Roll No']}
                </h4>
              </Col>
              <Col />
              <Col className="test_col">
                {arrange(data)}
                <CSVLink data={dataing} headers={headers}>
                  Download
                </CSVLink>;
              </Col>
            </Row>
          )
        })
      )
    }
  };

  const deleteIcon = async (id) => {
    try {
      console.log('HERE in the world')
      const config =  {headers : {
        "Content-Type": "multipart/form-data",
        }
      }
      const formData = new FormData()
      formData.append('id', id)
      const result = await axios.post('http://127.0.0.1:8000/deletetest', formData, config)
      // if (result.status == 200){
        setReload(!reload)
        getData()
      // }
    }
    catch (e) {
      console.log("erroring", e)
    }
  }

  const arrange = (data) => {
    headers = [
      { label: "Roll No", key: "Roll No" },
      { label: "Q1", key: "Q1" },
      { label: "Q1_feedback", key: "Q1_feedback" },
      { label: "Q2", key: "Q2" },
      { label: "Q2_feedback", key: "Q2_feedback" },
      { label: "Q3", key: "Q3" },
      { label: "Q3_feedback", key: "Q3_feedback" },
      { label: "Q4", key: "Q4" },
      { label: "Q4_feedback", key: "Q4_feedback" },
      { label: "Q5", key: "Q5" },
      { label: "Q5_feedback", key: "Q5_feedback" },
      { label: "Q6", key: "Q6" },
      { label: "Q6_feedback", key: "Q6_feedback" },
      { label: "Q7", key: "Q7" },
      { label: "Q7_feedback", key: "Q7_feedback" },
      { label: "Q8", key: "Q8" },
      { label: "Q8_feedback", key: "Q8_feedback" },
      { label: "Q9", key: "Q9" },
      { label: "Q9_feedback", key: "Q9_feedback" },
      { label: "Correct", key: "Correct" },
      { label: "Grade", key: "Grade" },
      { label: "Improvement_Feedback", key: "Improvement_Feedback" },
      { label: "Percentage", key: "Percentage" },
      { label: "Total", key: "Total" },
    ]

    dataing = [
      { "Roll No": data["Roll No"], 
      "Q1": data.Q1,
      "Q1_feedback": data.Q1_feedback,
      "Q2": data.Q2 ,
      "Q2_feedback": data.Q2_feedback ,
      "Q3": data.Q3 ,
      "Q3_feedback": data.Q3_feedback ,
      "Q4": data.Q4 ,
      "Q4_feedback": data.Q4_feedback ,
      "Q5": data.Q5 ,
      "Q5_feedback": data.Q5_feedback ,
      "Q6": data.Q6 ,
      "Q6_feedback": data.Q6_feedback ,
      "Q7": data.Q7 ,
      "Q7_feedback": data.Q7_feedback ,
      "Q8": data.Q8,
      "Q8_feedback": data.Q8_feedback ,
      "Q9": data.Q9 ,
      "Q9_feedback": data.Q9_feedback ,
      "Correct": data.Correct ,
      "Grade": data.Grade ,
      "Improvement_Feedback": improvementFeedback ,
      "Percentage": data.Percentage ,
      "Total": data.Total ,}
    ]
  }

  const getFeedback = async (id) => {
    try {
      setLoading(true)
      const config =  {headers : {
        "Content-Type": "multipart/form-data",
        }
      }
      const formData = new FormData()
      formData.append('pk', id)
      const result = await axios.post('http://127.0.0.1:8000/showtest', formData, config)
      const arr = []
      Object.keys(result.data).map(v => arr.push(result.data[v]))
      setFeedbackData(arr)   
    }
    catch (e) {
      console.log('SHOWTESR ERR', e)
    }  
    finally {
      setLoading(false)
    } 
 }

  useEffect(() => {
    getData()
  }, [feedbackData, ])


  const uploadFIle = async () => {
    if (name && number && file && gradefile){
      setErrorMsg('')
      const config =  {headers : {
        "Content-Type": "multipart/form-data",
        }
      }
      const formData = new FormData()
      formData.append('name', name)
      formData.append('no', number)
      formData.append('id', user_name)
      formData.append('test_file', file)
      formData.append('grades_file', gradefile)
      setLoading(true)
      axios.post('http://127.0.0.1:8000/addtest', formData, config ).then(res => {
        setReload(!reload)
        setLoading(false)
        getData()
      }).catch(err => {
        setLoading(false)
      })
    }
    else {
      setErrorMsg("Error! Something's missing, please fill all the required fields.")
    }
    
  }

  const getData = async () => {
    try {
      setLoading(true)
      const config =  {headers : {
        "Content-Type": "multipart/form-data",
        }
      }
      const formData = new FormData()
      formData.append('id', user_name)
      const result = await axios.post('http://127.0.0.1:8000/showtestlist', formData, config)
      setData(JSON.parse(result.data.result))
    }
    catch (e) {

    }
    finally {
      setLoading(false)
    }
  }

  const handleClose = () => setModal(false)
   
  return(
    <>
    {
      loading ?

      <div style={{display: 'flex'}}>
        <div style={{marginLeft: '49%', marginTop: '20%', justifyContent: 'center', alignContent: 'center', }}>
          <Spinner animation="border" role="status" style={{ fontSize: 40}}>
            <span className="sr-only">Loading...</span>
          </Spinner>
        </div>  
      </div>
      :
      
      <div>
        <Modal show={modal} onHide={handleClose}>
          <Modal.Header closeButton>
            <Modal.Title>Result</Modal.Title>
          </Modal.Header>
          <Modal.Body>
            {       
                     
              modalValues.values.slice(0, -6).map((data) => {
                return (
                  <>
                    <h6 
                      className={ (typeof(data) == 'number' && data < 0.5) ? 'modal_dataW20_red' : typeof(data) == 'number' ? 'modal_dataW20'  : 'modal_dataW80' }
                    >
                      {typeof(data) == 'number' ? data.toFixed(2) : data}
                    </h6>
                  </>
                )
              })
            }
          </Modal.Body>
          <Modal.Footer className='mr-auto modal_footer'>
            <div>
              <p style={{marginBottom: '0.2rem'}}><span style={{fontWeight: 'bold'}}>Total Score: </span>{footerValues.total}</p>
              <p style={{marginBottom: '0.2rem'}}><span style={{fontWeight: 'bold', color: 'green'}}>Correct Score: </span>{footerValues.correct}</p>
              <p style={{marginBottom: '0.2rem'}}><span style={{fontWeight: 'bold'}}>Percentage: </span>{footerValues.percentage}</p>
              <p style={{marginBottom: '0.2rem'}}><span style={{fontWeight: 'bold'}}>Grade: </span>{footerValues.grade}</p>
              <p style={{marginBottom: '0.2rem', width: '100%'}}><span style={{fontWeight: 'bold'}}>Feedback: </span>
                  <input 
                    style={{ border: 'none', width: '80%' }}
                    value={improvementFeedback}
                    onChange={ v => { 
                      setImprovementFeedback(v.target.value) 
                      footerValues.improvementFeedback = v.target.value
                    }}
                  />
                </p>
              <p style={{ marginTop: '0.4rem', marginBottom: '0.2rem', fontWeight: 'bold' }}>Word2Vec model is used with the accuracy of '74.07%'</p>
            </div>
          </Modal.Footer>
        </Modal>
        {/* <h1 style={{textAlign: 'center'}}>Welcome back! Here is a summary of your saved tests!</h1> */}
        <div className="test_details" style={{ borderRadius: 10 }}>
          <h2 className="title">
            { 
              feedbackData ? 
              <img 
                onClick={() => { 
                setFeedback(false)
                setFeedbackData(false)}} 
                style={{ width: '30px', height: '30px', marginRight: '5em' }} 
                src={ BackArrow } 
              /> : ''
            }
            {/* ExamApp - Giving Accurate feedback for assesments */}
          </h2>
          <div className="details_wrapper">
            <h2 className="details_title">
              {`Welcome back ${user_name}, heres a summary of your saved tests`}
            </h2>
            <Container fluid className="details_header" style={{ backgroundColor: '#001529', borderRadius: 10 }}>
              <Row>
                <Col className="header_title">
                  <h4 style={{color: '#fff'}}>{!feedbackData ? "Test Name" : "Student" }</h4>
                </Col>
                <Col className="header_title">
                  <h4 style={{color: '#fff'}}>{!feedbackData ? "Number" : ""}</h4>
                </Col>
                <Col className="header_title" />
              </Row>
            </Container>
            <Container className="test_container" fluid>
              {renderTestDetails()}
            </Container>
            { !feedbackData ?
              <Container className='upload_container' >
                  <h6 style={{ color: '#a6a6a6', fontSize: '18px' }}>upload your files</h6>
                  <img src={Downloadpicture} style={{ height: '40px', width: '40px'}} onClick={() => {uploadFIle()}}/>
                  <div style={{display: 'flex', justifyContent: 'space-around', marginBottom: '10px'  }} > 
                    <input 
                      className='input'
                      placeholder= 'name'
                      value = { name }
                      onChange={ v => setName(v.target.value) }
                    />
                    <input 
                      className='input'
                      placeholder= 'number'
                      value= { number }
                      onChange= { v => setNumber(v.target.value) }
                    />
                  </div> 
                  <div style={{display: 'flex', justifyContent: 'space-around',   }} > 
                  <Files
                    className='files-dropzone'
                    onChange={ e => setFile(e[0])}
                    // onError={this.onFilesError}
                    accepts={['image/png', '.pdf', 'audio/*', '.zip', '.csv']}
                    multiple
                    maxFileSize={10000000}
                    minFileSize={0}
                    clickable
                  >
                    <button className='btns'>Question Answers</button>
                  </Files>
                  <Files
                    className='files-dropzone'
                    onChange={ e => setGradeFile(e[0])}
                    // onError={this.onFilesError}
                    accepts={['image/png', '.pdf', 'audio/*', '.zip', '.csv']}
                    multiple
                    maxFileSize={10000000}
                    minFileSize={0}
                    clickable
                  >
                    <button className='btns'>Student answers</button>
                  </Files>     
                  </div>
              </Container>
              : 
              ''
            }
            {
                !feedbackData ?
                  errormsg ? 
                    <p style={{color: 'red'}}>{errormsg}</p>
                  :
                  ''
                  : 
                  ''
            }
          </div>
          
        </div>
      </div>
    }
    </>
  )
}
// }
export default SavedTests