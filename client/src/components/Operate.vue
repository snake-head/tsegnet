<template>
  <div>
      <el-button type="primary" @click="predCentroids">预测质心</el-button>
      <el-button type="primary" @click="predCentroids2">预测质心(预分割)</el-button>
      <el-button type="primary" @click="trueCentroids">真实质心</el-button>
      <br>
      <el-button type="primary" @click="predSeg">分割</el-button>
      <el-button type="primary">导出</el-button>
  </div>
</template>

<script>
import $store from '../store/index'
import qs from 'qs'
import axios from 'axios'
import { reactive } from 'vue'
export default {
    name:'Operate',
    setup() {
      let modelData = reactive({
        points: [],
        normal: []
      })
      function predCentroids(){
        if (!$store.state.currentUrl){
          alert('未选择模型')
        }
        axios.post($store.state.currentUrl, qs.stringify({
          operate: 'predCentroids',
        }),{
          'Content-Type': 'application/json; charset=utf-8'
        }).then(
          response=>{
            if (response.data){
              $store.dispatch('setCentroidsPred', (JSON.parse(response.data.data)).centroidsPred)
              $store.dispatch('displayCentroids', true)
              modelData.points = JSON.parse(response.data.data).points
              modelData.normal = JSON.parse(response.data.data).normal
            }
            else{
              console.log(response.data.status)
            }
          },
          error=>{
            console.log('error', error.message)
          }
        )
      }

      function predSeg(){
        if (!$store.state.currentUrl){
          alert('未选择模型')
        }
        if (!$store.state.ifDisplayCentroids){
          alert('未预测质心')
        }
        axios.post($store.state.currentUrl, qs.stringify({
          operate: 'predSeg',
          centroidsPred: ($store.state.centroidsPred).toString(),
          points: (modelData.points).toString(),
          normal: (modelData.normal).toString()
        }),{
          'Content-Type': 'application/json; charset=utf-8'
        }).then(
          response=>{
            if (response.data){
              // $store.dispatch('setSceneData', (JSON.parse(response.data.data)).sceneData)
              // $store.dispatch('setModelColor', (JSON.parse(response.data.data)).modelColor)
              $store.dispatch('displayResult', true)
            }
            else{
              console.log(response.data.status)
            }
          },
          error=>{
            console.log('error', error.message)
          }
        )

      }

      function predCentroids2(){
        if (!$store.state.currentUrl){
          alert('未选择模型')
        }
        axios.post($store.state.currentUrl, qs.stringify({
          operate: 'predCentroids2',
        }),{
          'Content-Type': 'application/json; charset=utf-8'
        }).then(
          response=>{
            if (response.data){
              $store.dispatch('setCentroidsPred', (JSON.parse(response.data.data)).centroidsPred)
              $store.dispatch('displayCentroids', true)
              modelData.points = JSON.parse(response.data.data).points
              modelData.normal = JSON.parse(response.data.data).normal
            }
            else{
              console.log(response.data.status)
            }
          },
          error=>{
            console.log('error', error.message)
          }
        )
      }

      function trueCentroids(){
        if (!$store.state.currentUrl){
          alert('未选择模型')
        }
        axios.post($store.state.currentUrl, qs.stringify({
          operate: 'trueCentroids',
        }),{
          'Content-Type': 'application/json; charset=utf-8'
        }).then(
          response=>{
            if (response.data){
              $store.dispatch('setCentroidsPred', (JSON.parse(response.data.data)).centroidsPred)
              $store.dispatch('displayCentroids', true)
              modelData.points = JSON.parse(response.data.data).points
              modelData.normal = JSON.parse(response.data.data).normal
            }
            else{
              console.log(response.data.status)
            }
          },
          error=>{
            console.log('error', error.message)
          }
        )
      }
      
      return {
        predCentroids,
        predCentroids2,
        predSeg,
        trueCentroids,
        modelData
      }
    }
}
</script>

<style>
  .el-button{
    margin: 10px;
  }
</style>